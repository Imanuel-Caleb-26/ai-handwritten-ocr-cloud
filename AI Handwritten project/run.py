"""
Server entry point for the AI Handwritten OCR application.

Usage:
    python run.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UPLOAD_FOLDER, OUTPUT_FOLDER

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI HANDWRITTEN — OCR Server (Cloud Edition)")
    print("="*55)

    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\n  Pre-loading Cloud OCR engine (may take a moment)...")

    try:
        from src.ocr_engine import get_engine
        engine = get_engine()
        info = engine.model_info()
        print(f"  Model loaded:  [{info['label']}]  on {info['device'].upper()}")
    except Exception as e:
        print(f"\n  ✗ CRITICAL ERROR during model load: {e}")
        sys.exit(1)

    print("\n  Server is LIVE at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.\n")
    print("="*55 + "\n")

    from app import app
    app.run(host="127.0.0.1", port=5000, debug=False)
