"""
Iterative training runner - trains, evaluates, and saves best model.
Re-run this script to keep improving the model over multiple sessions.

Usage:
    python scripts/run_training.py                         # Full training
    python scripts/run_training.py --epochs 1 --max-train 500  # Quick test
    python scripts/run_training.py --eval-only             # Evaluate existing model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable  # Use the same Python interpreter

def run_step(cmd_args: list, label: str) -> int:
    """Run a subprocess step and stream output."""
    print(f"\n{'='*60}")
    print(f"  ▶ {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd_args, cwd=str(PROJECT_ROOT))
    return result.returncode

def main(args):
    print("\n" + "█"*60)
    print("  AI HANDWRITTEN — ITERATIVE TRAINING PIPELINE")
    print("█"*60)

    model_path = PROJECT_ROOT / "models" / "trocr-iam-finetuned"

    if model_path.exists():
        print(f"\n  ℹ  Existing model found at: {model_path}")
        with open(model_path / "training_summary.txt") as f:
            print(f.read())
        print("  Training will CONTINUE from this checkpoint (add --restart to start fresh)")
    else:
        print("\n  ℹ  No existing model. Starting from microsoft/trocr-large-handwritten base.")

    if args.eval_only:
        # Only evaluate
        eval_cmd = [PYTHON, "training/evaluate_model.py", "--max-samples", "300"]
        rc = run_step(eval_cmd, "Evaluating Model")
        sys.exit(rc)

    # ── Step 1: Train ──
    train_cmd = [PYTHON, "training/train.py"]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        train_cmd += ["--batch-size", str(args.batch_size)]
    if args.max_train:
        train_cmd += ["--max-train", str(args.max_train)]
    if args.max_val:
        train_cmd += ["--max-val", str(args.max_val)]
    if args.restart:
        train_cmd += ["--restart"]

    rc = run_step(train_cmd, "Training TrOCR Model")
    if rc != 0:
        print(f"\n  ✗ Training failed with code {rc}. Check logs above.")
        sys.exit(rc)

    # ── Step 2: Evaluate ──
    if not args.skip_eval:
        eval_cmd = [
            PYTHON, "training/evaluate_model.py",
            "--model-path", str(model_path),
            "--max-samples", str(args.eval_samples),
        ]
        rc = run_step(eval_cmd, "Evaluating Fine-Tuned Model")
        if rc != 0:
            print("  ⚠  Evaluation failed, but model was saved. You can run evaluate manually.")

    print("\n" + "█"*60)
    print("  ✔ PIPELINE COMPLETE")
    print(f"  Model: {model_path}")
    print("  Re-run this script to keep training and improving accuracy!")
    print("█"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative TrOCR training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full training (3 epochs, all data):
    python scripts/run_training.py

  Quick test (1 epoch, 500 samples):
    python scripts/run_training.py --epochs 1 --max-train 500 --eval-samples 100

  Continue training 2 more epochs:
    python scripts/run_training.py --epochs 2

  Evaluate existing model only:
    python scripts/run_training.py --eval-only

  Start training from scratch:
    python scripts/run_training.py --restart
        """,
    )
    parser.add_argument("--epochs",       type=int,  default=None,  help="Training epochs")
    parser.add_argument("--batch-size",   type=int,  default=None,  help="Batch size")
    parser.add_argument("--max-train",    type=int,  default=None,  help="Max training samples")
    parser.add_argument("--max-val",      type=int,  default=None,  help="Max validation samples")
    parser.add_argument("--eval-samples", type=int,  default=300,   help="Samples for evaluation")
    parser.add_argument("--skip-eval",    action="store_true",       help="Skip evaluation after training")
    parser.add_argument("--eval-only",    action="store_true",       help="Skip training, only evaluate")
    parser.add_argument("--restart",      action="store_true",       help="Ignore checkpoint, restart training")
    args = parser.parse_args()
    main(args)
