# DocLingua - Bilingual Document OCR & NER Pipeline

![Project Banner](https://img.shields.io/badge/BilingualDocOCR%20Document%20AI-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

An end-to-end AI pipeline for processing bilingual (Nepali/English) documents featuring automated layout detection, language-aware OCR, and entity extraction.

## ✨ Features

- **📄 Document Layout Analysis**: YOLOv8-based detection of document regions
- **🌐 Bilingual OCR**: Intelligent language detection for Nepali and English documents
- **🏷️ Entity Extraction**: Rule-based entity extraction
- **🎯 Automatic Language Detection**: Based on document layout features
- **🌐 Web Interface**: Streamlit-based interactive application

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/manjil-budhathoki/BilingualDocOCR
cd BilingualDocOCR

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run src/app.py
```

Then open your browser at `http://localhost:8501`

## 🏗️ How It Works

1. **Upload** a document image
2. **YOLO detects** document regions (text blocks, photos, fingerprints)
3. **System determines** language:
   - Photo region → Nepali document (front side)
   - Fingerprint region → English document (back side)
4. **OCR processes** text using appropriate engine
5. **Entities are extracted** from the text
6. **Results are displayed** in the web interface

## 📁 Project Structure

```
DocLingua/
├── src/                    # Source code
│   ├── app.py            # Main application
│   ├── NER/              # Entity extraction
│   └── OCR/              # OCR processing
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Streamlit](https://streamlit.io/)

## 📧 Contact

Your Name - manjilbbudhathoki@gmail.com

Project Link: [https://github.com/manjil-budhathoki/BilingualDocOCR](https://github.com/manjil-budhathoki/BilingualDocOCR)