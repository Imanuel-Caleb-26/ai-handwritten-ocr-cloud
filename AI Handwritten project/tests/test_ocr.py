"""
Clean OCR engine test — verifies the engine loads without errors.
Run: python -m pytest tests/test_ocr.py -v
"""
import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config_imports():
    """Config module should import cleanly."""
    import config
    assert hasattr(config, "BASE_MODEL_NAME")
    assert hasattr(config, "FINETUNED_MODEL_DIR")
    assert hasattr(config, "ALLOWED_EXTENSIONS")


def test_image_processor_imports():
    """image_processor should import without errors."""
    from src.utils.image_processor import preprocess_image, get_line_segments
    assert callable(preprocess_image)
    assert callable(get_line_segments)


def test_doc_generator_imports():
    """doc_generator should import without errors."""
    from src.utils.doc_generator import generate_docx, generate_pdf
    assert callable(generate_docx)
    assert callable(generate_pdf)


def test_app_imports():
    """Flask app should import without errors."""
    from app import app
    assert app is not None
    assert app.name == "app"


@pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_TEST"),
    reason="Set RUN_MODEL_TEST=1 to load model (slow)",
)
def test_engine_loads():
    """OCR engine should load and report model info."""
    from src.ocr_engine import OCREngine
    engine = OCREngine()
    info = engine.model_info()
    assert "is_finetuned" in info
    assert "device" in info
    assert "label" in info
