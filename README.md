# AI Handwritten Recognition to System Files (Cloud Edition)

A professional web application that converts handwritten notes into editable Word (.docx) and PDF documents using state-of-the-art **Cloud OCR APIs**. 

## Features
- **High Accuracy**: Uses Google Gemini (1.5 Flash) with an automated fallback to OCR Space for robust text extraction.
- **Lightweight**: Purely cloud-based inference means no heavy local ML libraries (like PyTorch or Transformers) and no large model downloads.
- **Modern UI**: Dark-themed, responsive interface with drag-and-drop support.
- **Dual Export**: Download results as professional Word and PDF files.


## Tech Stack
- **Backend**: Flask (Python)
- **AI Models**: Google Gemini (Cloud), OCR Space (Cloud)
- **Image Processing**: PIL
- **Doc Generation**: `python-docx`, `fpdf2`
- **Frontend**: Vanilla HTML/CSS/JS (Modern Glassmorphism)
