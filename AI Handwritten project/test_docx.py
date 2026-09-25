import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.utils.doc_generator import generate_docx

try:
    out_path = os.path.join(PROJECT_ROOT, "test_out.docx")
    generate_docx("Hello\n\nWorld", out_path)
    print("Docx generation successful.")
except Exception as e:
    print("Docx generation failed!")
    traceback.print_exc()
