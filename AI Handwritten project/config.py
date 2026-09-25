"""
Central configuration for the AI Handwritten OCR project.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER    = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER    = os.path.join(BASE_DIR, "outputs")
STATIC_DEBUG_DIR = os.path.join(BASE_DIR, "static", "debug")

# --- Allowed upload extensions ---
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# --- Debug ---
DEBUG_MODE = True

# --- API Keys ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "helloworld")
