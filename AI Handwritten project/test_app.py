import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app import app
from flask import json

with app.test_client() as client:
    # We will just post an image to /upload
    img_path = os.path.join(PROJECT_ROOT, "uploads", "0c963e40-acbb-4adc-8542-be82eba268b3.png")
    with open(img_path, "rb") as img:
        response = client.post('/upload', data={"image": img})
        data = json.loads(response.data)
        
        print("Upload Response:", response.status_code)
        print("Text:", data.get('text'))
        print("Error:", data.get('error'))

        session_id = data.get('id')
        if session_id:
            res_docx = client.get(f"/download/docx/{session_id}")
            print("Docx Download:", res_docx.status_code)
            
            res_pdf = client.get(f"/download/pdf/{session_id}")
            print("Pdf Download:", res_pdf.status_code)
