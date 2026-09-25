"""
Flask web application for the AI Handwritten OCR project.
Routes: / (index), /upload, /download/<format>/<id>, /status
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS
from src.ocr_engine import get_engine
from src.utils.doc_generator import generate_docx, generate_pdf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# In-memory text cache keyed by session ID
text_cache: dict = {}


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    """Returns model status for the UI badge."""
    try:
        model_type = request.args.get("model", "pipeline")
        engine = get_engine(model_type)
        info = engine.model_info()
        return jsonify({"ok": True, **info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save uploaded file with a unique ID
    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    img_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
    file.save(img_path)

    model_type = request.form.get("model", "pipeline")

    try:
        print(f"Starting OCR for {img_path} with model {model_type}…")
        engine = get_engine(model_type)
        text = engine.recognize_text(img_path)

        if not text.strip():
            text = "[No text detected — try a clearer image or higher contrast]"

        text_cache[file_id] = text
        print(f"OCR done. Session ID: {file_id}")

        return jsonify({
            "id":     file_id,
            "text":   text,
            "model":  engine.model_info()["label"],
        })

    except Exception as e:
        print(f"OCR Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/update_text", methods=["POST"])
def update_text():
    data = request.get_json()
    if data and "id" in data and "text" in data:
        text_cache[data["id"]] = data["text"]
    return jsonify({"success": True})

@app.route("/download/<fmt>/<session_id>")
def download(fmt: str, session_id: str):
    if session_id not in text_cache:
        return "Session expired or not found", 404

    text = text_cache[session_id]

    if fmt == "docx":
        out_path = os.path.join(OUTPUT_FOLDER, f"{session_id}.docx")
        generate_docx(text, out_path)
        return send_file(out_path, as_attachment=True, download_name="handwriting_output.docx")

    elif fmt == "pdf":
        out_path = os.path.join(OUTPUT_FOLDER, f"{session_id}.pdf")
        generate_pdf(text, out_path)
        return send_file(out_path, as_attachment=True, download_name="handwriting_output.pdf")

    return "Invalid format — use 'docx' or 'pdf'", 400
