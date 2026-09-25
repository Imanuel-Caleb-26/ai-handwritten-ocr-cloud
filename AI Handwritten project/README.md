# AI Handwritten Recognition to System Files (Cloud Edition)

A professional web application that converts handwritten notes into editable Word (.docx) and PDF documents using state-of-the-art **Cloud OCR APIs**. 

## Features
- **High Accuracy**: Uses Google Gemini (1.5 Flash) with an automated fallback to OCR Space for robust text extraction.
- **Lightweight**: Purely cloud-based inference means no heavy local ML libraries (like PyTorch or Transformers) and no large model downloads.
- **Modern UI**: Dark-themed, responsive interface with drag-and-drop support.
- **Dual Export**: Download results as professional Word and PDF files.

## Project Structure
```text
AI Handwritten project/
├── app.py              # Flask web application
├── run.py              # Server startup script
├── requirements.txt    # Python dependencies
├── README.md
├── src/                # Core source code
│   ├── ocr_engine.py       # Cloud OCR Pipeline (Gemini -> OCR Space)
│   └── utils/
│       └── doc_generator.py    # DOCX/PDF generation
├── static/             # CSS & JS assets
├── templates/          # HTML templates
├── uploads/            # Uploaded images (auto-created)
└── outputs/            # Generated documents (auto-created)
```

## Prerequisites
- Python 3.8+

## Installation

1. **Clone or Download** this project to your local machine.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Keys**:
   Set up these environment variables to enable the cloud OCR pipeline:
   - `GEMINI_API_KEY`: Your Google Gemini API Key (Primary Engine)
   - `OCR_SPACE_API_KEY`: Your OCR Space API Key (Fallback, defaults to `helloworld`)

## Usage

1. **Start the Server**:
   ```bash
   python run.py
   ```
2. **Open Browser**: Go to `http://localhost:5000`.
3. **Upload**: Drag an image of handwriting onto the upload zone. The app will automatically process it using Gemini, falling back to OCR Space if needed.
4. **Download**: Once processed, use the buttons to export your text.

## Tech Stack
- **Backend**: Flask (Python)
- **AI Models**: Google Gemini (Cloud), OCR Space (Cloud)
- **Image Processing**: PIL
- **Doc Generation**: `python-docx`, `fpdf2`
- **Frontend**: Vanilla HTML/CSS/JS (Modern Glassmorphism)
