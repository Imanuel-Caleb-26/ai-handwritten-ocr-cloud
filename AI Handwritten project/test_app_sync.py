import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app import app
from flask import json

with app.test_client() as client:
    # 1. We post an image to /upload
    img_path = os.path.join(PROJECT_ROOT, "uploads", "0c963e40-acbb-4adc-8542-be82eba268b3.png")
    with open(img_path, "rb") as img:
        response = client.post('/upload', data={"image": img})
        data = json.loads(response.data)
        
        session_id = data.get('id')
        print("Upload Text:", data.get('text'))
        
        if session_id:
            # 2. We update the text cache to simulate user editing
            edited_text = "This is a custom edit done by the user.\n\nIt has blank lines!"
            client.post('/update_text', json={"id": session_id, "text": edited_text})
            
            # 3. Download DOCX
            res_docx = client.get(f"/download/docx/{session_id}")
            print("Docx Download:", res_docx.status_code)
            
            # 4. Download PDF
            res_pdf = client.get(f"/download/pdf/{session_id}")
            print("Pdf Download:", res_pdf.status_code)

            # Let's save them locally to verify
            with open("test_out_edited.docx", "wb") as f:
                f.write(res_docx.data)
            with open("test_out_edited.pdf", "wb") as f:
                f.write(res_pdf.data)
            
            print("Successfully saved DOCX and PDF with edited text!")
