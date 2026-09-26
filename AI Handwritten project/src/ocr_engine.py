"""
OCR Engine — Uses Gemini OCR and OCR Space (Cloud APIs only).
"""

import os
import sys
from pathlib import Path
from PIL import Image
import requests
import google.generativeai as genai

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GEMINI_API_KEY,
    OCR_SPACE_API_KEY,
)

class GeminiEngine:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def model_info(self) -> dict:
        return {
            "is_finetuned": False,
            "device": "cloud",
            "label": "Gemini OCR",
        }

    def recognize_text(self, image_path: str) -> str:
        print(f"Processing (Gemini): {image_path}")
        img = Image.open(image_path)
        prompt = "Extract all handwritten text from this image exactly as written. Return only the text without markdown formatting or other comments."
        response = self.model.generate_content([prompt, img])
        return response.text.strip()


class OCRSpaceEngine:
    def model_info(self) -> dict:
        return {
            "is_finetuned": False,
            "device": "cloud",
            "label": "OCR Space",
        }

    def recognize_text(self, image_path: str) -> str:
        print(f"Processing (OCR Space): {image_path}")
        payload = {
            'isOverlayRequired': False,
            'apikey': OCR_SPACE_API_KEY,
            'language': 'eng',
            'isHandprinted': True,
        }
        with open(image_path, 'rb') as f:
            r = requests.post('https://api.ocr.space/parse/image',
                              files={image_path: f},
                              data=payload,
                              timeout=30)
        data = r.json()
        if data.get("IsErroredOnProcessing"):
            raise Exception(data.get("ErrorMessage", ["Unknown OCR error"])[0])
        parsed_results = data.get("ParsedResults", [])
        if not parsed_results:
            return ""
        return parsed_results[0].get("ParsedText", "").strip()


class EasyOCREngine:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False)

    def model_info(self) -> dict:
        return {
            "is_finetuned": False,
            "device": "local",
            "label": "EasyOCR",
        }

    def recognize_text(self, image_path: str) -> str:
        print(f"Processing (EasyOCR): {image_path}")
        results = self.reader.readtext(image_path)
        results_sorted = sorted(results, key=lambda item: item[0][0][1])
        text_lines = [text for bbox, text, prob in results_sorted]
        return "\n".join(text_lines)


class PipelineEngine:
    def __init__(self):
        self.gemini = GeminiEngine()
        self.ocrspace = OCRSpaceEngine()
        self.easyocr = EasyOCREngine()

    def model_info(self) -> dict:
        return {
            "is_finetuned": False,
            "device": "hybrid",
            "label": "Pipeline (Gemini -> OCR Space -> EasyOCR)",
        }

    def recognize_text(self, image_path: str) -> str:
        try:
            print("Pipeline: Trying Gemini OCR...")
            text = self.gemini.recognize_text(image_path)
            if text:
                return text
        except Exception as e:
            print(f"Pipeline: Gemini failed - {e}")

        try:
            print("Pipeline: Trying OCR Space...")
            text = self.ocrspace.recognize_text(image_path)
            if text:
                return text
        except Exception as e:
            print(f"Pipeline: OCR Space failed - {e}")

        try:
            print("Pipeline: Trying EasyOCR (Local)...")
            text = self.easyocr.recognize_text(image_path)
            if text:
                return text
        except Exception as e:
            print(f"Pipeline: EasyOCR failed - {e}")
            raise Exception("All OCR models (Cloud & Local) failed.")


# ─── Singleton ────────────────────────────────────────────────────────────────
_engines = {}

def get_engine(model_type: str = "pipeline"):
    global _engines
    if model_type not in _engines:
        if model_type == "gemini":
            _engines[model_type] = GeminiEngine()
        elif model_type == "ocrspace":
            _engines[model_type] = OCRSpaceEngine()
        elif model_type == "easyocr":
            _engines[model_type] = EasyOCREngine()
        else:
            _engines[model_type] = PipelineEngine()
    return _engines[model_type]


