"""
Training script for fine-tuning TrOCR on the IAM Handwriting Database.

Usage:
    python training/train.py
    python training/train.py --epochs 5 --batch-size 2
    python training/train.py --max-train 1000  # quick test with 1000 samples
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric
from torch.utils.data import Dataset
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "training" / "config.yaml"
MODELS_DIR   = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "training" / "logs"

# ─── Load config ──────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ─── Dataset wrapper ──────────────────────────────────────────────────────────
class IAMDataset(Dataset):
    """Wraps HuggingFace IAM dataset for TrOCR fine-tuning."""

    def __init__(self, hf_dataset, processor, max_target_length=128):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Image
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to standard height while keeping aspect ratio
        target_h = CFG["image"]["height"]
        w, h = image.size
        scale = target_h / h
        new_w = min(int(w * scale), CFG["image"]["max_width"])
        image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Process text label
        text = item.get("text", item.get("transcription", ""))
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Replace padding with -100 so loss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ─── CER Metric ───────────────────────────────────────────────────────────────
def build_compute_metrics(processor):
    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        label_ids   = pred.label_ids
        pred_ids    = pred.predictions

        # Replace -100 with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics


# ─── Main training function ────────────────────────────────────────────────────
def train(args):
    print("\n" + "="*60)
    print("  TrOCR Fine-Tuning on IAM Handwriting Database")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")
    if device == "cpu":
        print("  ⚠  CPU training is slow. Expect ~1-2 hrs/epoch.")
        print("     To speed up: use --max-train 500 for a quick test run.")

    # ── Output directory ──
    output_dir = PROJECT_ROOT / CFG["model"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load processor & model ──
    base_model = CFG["model"]["base"]
    print(f"\n  Loading base model: {base_model}")
    processor = TrOCRProcessor.from_pretrained(base_model)

    # Load from checkpoint if fine-tuned model already exists
    if (output_dir / "config.json").exists() and not args.restart:
        print(f"  Found existing checkpoint at {output_dir}. Continuing training...")
        model = VisionEncoderDecoderModel.from_pretrained(str(output_dir))
    else:
        print("  Starting from base model (no existing checkpoint or --restart set).")
        model = VisionEncoderDecoderModel.from_pretrained(base_model)

    # ── Configure model generation settings ──
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = CFG["generation"]["max_new_tokens"]
    model.config.early_stopping         = CFG["generation"]["early_stopping"]
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = CFG["generation"]["num_beams"]

    model.to(device)

    # ── Load dataset ──
    dataset_name = CFG["dataset"]["name"]
    print(f"\n  Loading dataset: {dataset_name}")
    print("  (Downloading ~2GB on first run — please wait...)\n")

    try:
        raw = load_dataset(dataset_name)
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        print("  Trying fallback: IAM lines from another source...")
        # Fallback to alternative IAM source
        raw = load_dataset("Teklia/IAM-line", trust_remote_code=True)

    train_ds = raw[CFG["dataset"]["train_split"]]
    val_ds   = raw[CFG["dataset"]["validation_split"]]

    # Limit samples for quick testing
    max_train = args.max_train or CFG["dataset"]["max_train_samples"]
    max_val   = args.max_val  or CFG["dataset"]["max_val_samples"]

    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_val:
        val_ds = val_ds.select(range(min(max_val, len(val_ds))))

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")

    # ── Wrap in torch Dataset ──
    train_dataset = IAMDataset(train_ds, processor)
    val_dataset   = IAMDataset(val_ds,   processor)

    # ── Training arguments ──
    num_epochs  = args.epochs or CFG["training"]["num_epochs"]
    batch_size  = args.batch_size or CFG["training"]["batch_size"]
    run_name    = f"trocr-iam-{datetime.now().strftime('%Y%m%d_%H%M')}"

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate          = True,
        evaluation_strategy            = "steps",
        save_strategy                  = "steps",
        per_device_train_batch_size    = batch_size,
        per_device_eval_batch_size     = batch_size,
        num_train_epochs               = num_epochs,
        learning_rate                  = CFG["training"]["learning_rate"],
        weight_decay                   = CFG["training"]["weight_decay"],
        warmup_steps                   = CFG["training"]["warmup_steps"],
        fp16                           = CFG["training"]["fp16"] and device == "cuda",
        logging_steps                  = CFG["training"]["logging_steps"],
        eval_steps                     = CFG["training"]["eval_steps"],
        save_steps                     = CFG["training"]["save_steps"],
        output_dir                     = str(LOG_DIR),
        overwrite_output_dir           = True,
        save_total_limit               = 3,
        load_best_model_at_end         = True,
        metric_for_best_model          = "cer",
        greater_is_better              = False,  # lower CER = better
        gradient_accumulation_steps    = CFG["training"]["gradient_accumulation_steps"],
        max_grad_norm                  = CFG["training"]["max_grad_norm"],
        dataloader_num_workers         = CFG["training"]["dataloader_num_workers"],
        run_name                       = run_name,
        report_to                      = "none",  # disable wandb etc
    )

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        model           = model,
        tokenizer       = processor.feature_extractor,
        args            = training_args,
        compute_metrics = build_compute_metrics(processor),
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = default_data_collator,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Run training ──
    print(f"\n  Starting training: {num_epochs} epoch(s), batch size {batch_size}")
    print(f"  Effective batch size: {batch_size * CFG['training']['gradient_accumulation_steps']}")
    print("─"*60)

    trainer.train()

    # ── Save best model ──
    print(f"\n  Saving fine-tuned model to: {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # ── Final evaluation ──
    print("\n  Running final evaluation...")
    metrics = trainer.evaluate()
    cer = metrics.get("eval_cer", "N/A")
    print(f"\n  ✔ Training complete!")
    print(f"  Final CER (Character Error Rate): {cer:.4f}" if isinstance(cer, float) else f"  CER: {cer}")
    print(f"  Model saved to: {output_dir}")
    print("="*60 + "\n")

    # Save training summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Training Date: {datetime.now().isoformat()}\n")
        f.write(f"Base Model: {base_model}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Train Samples: {len(train_ds)}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Final CER: {cer}\n")
    print(f"  Summary saved to: {summary_path}")

    return str(output_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on IAM Handwriting dataset")
    parser.add_argument("--epochs",      type=int,   default=None, help="Override training epochs")
    parser.add_argument("--batch-size",  type=int,   default=None, help="Batch size per device")
    parser.add_argument("--max-train",   type=int,   default=None, help="Max training samples (for quick tests)")
    parser.add_argument("--max-val",     type=int,   default=None, help="Max validation samples")
    parser.add_argument("--restart",     action="store_true",      help="Ignore existing checkpoint and restart")
    args = parser.parse_args()

    train(args)
