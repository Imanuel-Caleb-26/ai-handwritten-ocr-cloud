"""
Evaluation script - computes CER and WER on the IAM test split.

Usage:
    python training/evaluate.py
    python training/evaluate.py --model-path models/trocr-iam-finetuned
    python training/evaluate.py --max-samples 200
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import load as load_metric
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def evaluate(model_path: str, max_samples: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print("  TrOCR Model Evaluation")
    print(f"{'='*55}")
    print(f"  Model  : {model_path}")
    print(f"  Device : {device.upper()}")

    # ── Load model & processor ──
    print("\n  Loading model...")
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model     = VisionEncoderDecoderModel.from_pretrained(model_path)
    except Exception as e:
        print(f"  ✗ Could not load model from {model_path}: {e}")
        return None

    model.to(device)
    model.eval()

    # ── Load test dataset ──
    print("  Loading IAM test split...")
    try:
        raw = load_dataset("Teklia/IAM-line")
        test_ds = raw["test"]
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return None

    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    print(f"  Test samples: {len(test_ds)}")

    # ── Metrics ──
    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")

    all_preds  = []
    all_labels = []

    print("\n  Running inference...\n")
    for item in tqdm(test_ds, desc="  Evaluating"):
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        target_h = 384
        w, h = image.size
        scale = target_h / h
        new_w = min(int(w * scale), 2500)
        image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

        inputs = processor(image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(
                inputs,
                max_new_tokens=128,
                num_beams=5,
            )
        pred_text  = processor.batch_decode(generated, skip_special_tokens=True)[0]
        label_text = item.get("text", item.get("transcription", ""))

        all_preds.append(pred_text)
        all_labels.append(label_text)

    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    wer = wer_metric.compute(predictions=all_preds, references=all_labels)

    print(f"\n  {'─'*45}")
    print(f"  Results on {len(test_ds)} test samples:")
    print(f"  CER (Character Error Rate): {cer:.4f}  ({'✔ Excellent' if cer < 0.1 else '◉ Good' if cer < 0.2 else '✗ Needs more training'})")
    print(f"  WER (Word Error Rate)     : {wer:.4f}  ({'✔ Excellent' if wer < 0.2 else '◉ Good' if wer < 0.4 else '✗ Needs more training'})")
    print(f"  {'─'*45}")

    # Sample predictions
    print("\n  Sample predictions (first 5):")
    for i in range(min(5, len(all_preds))):
        print(f"  [{i+1}] Pred  : {all_preds[i]}")
        print(f"       Label : {all_labels[i]}")
        print()

    # Save results
    results_path = PROJECT_ROOT / "models" / "evaluation_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Samples: {len(test_ds)}\n")
        f.write(f"CER: {cer:.4f}\n")
        f.write(f"WER: {wer:.4f}\n\n")
        f.write("Sample Predictions:\n")
        for i in range(min(10, len(all_preds))):
            f.write(f"Pred  : {all_preds[i]}\n")
            f.write(f"Label : {all_labels[i]}\n\n")

    print(f"  Results saved to: {results_path}")
    print(f"{'='*55}\n")
    return {"cer": cer, "wer": wer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TrOCR model on IAM test set")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "trocr-iam-finetuned"),
        help="Path to model directory (default: models/trocr-iam-finetuned)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples")
    args = parser.parse_args()

    evaluate(args.model_path, args.max_samples)
