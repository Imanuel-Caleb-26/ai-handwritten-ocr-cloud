import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.ocr_engine import get_engine

try:
    engine = get_engine()
    img_path = os.path.join(PROJECT_ROOT, "uploads", "0c963e40-acbb-4adc-8542-be82eba268b3.png")
    text = engine.recognize_text(img_path)
    print("\n--- RECOGNIZED TEXT ---")
    print(text)
    print("-----------------------\n")
except Exception as e:
    import traceback
    traceback.print_exc()
