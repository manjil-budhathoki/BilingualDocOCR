import cv2
import tempfile
from language_detector import detect_language_from_regions  # NEW IMPORT
from OCR.Main_ocr import preprocess
from NER.labeler.weak_labeler import WeakLabeler

# ===============================
# GLOBAL SINGLETONS (IMPORTANT)
# ===============================

_easy_reader = None
_doctr_model = None
_labeler = None


# ===============================
# LOADERS
# ===============================

def _load_easyocr():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        _easy_reader = easyocr.Reader(["ne", "en"], gpu=False)
    return _easy_reader


def _load_doctr():
    global _doctr_model
    if _doctr_model is None:
        from doctr.models import ocr_predictor
        _doctr_model = ocr_predictor(pretrained=True)
    return _doctr_model


def _load_labeler():
    global _labeler
    if _labeler is None:
        _labeler = WeakLabeler()
    return _labeler


# ===============================
# OCR ROUTERS
# ===============================

def _ocr_english(processed_img):
    """
    English → DocTR (primary) with EasyOCR fallback
    """
    try:
        from doctr.io import DocumentFile

        model = _load_doctr()

        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            cv2.imwrite(tmp.name, processed_img)
            doc = DocumentFile.from_images(tmp.name)

        result = model(doc)

        texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [
                        w.value for w in line.words
                        if w.confidence > 0.3
                    ]
                    if words:
                        texts.append(" ".join(words))

        text = " ".join(texts).strip()

        if text:
            return text, "doctr"

    except Exception:
        pass

    # ---- fallback ----
    reader = _load_easyocr()
    text = " ".join(reader.readtext(processed_img, detail=0)).strip()
    return text, "easyocr_fallback"


def _ocr_nepali(processed_img):
    reader = _load_easyocr()
    text = " ".join(reader.readtext(processed_img, detail=0)).strip()
    return text, "easyocr"


# ===============================
# MAIN PIPELINE ENTRY
# ===============================

def process_image(image, detections, language="auto"):
    """
    image: full cv2 image
    detections: YOLO detections
    language: "auto" | "en" | "ne"
    """

    labeler = _load_labeler()

    collected_text = []
    engines_used = set()
    
    # NEW: Auto-detect language from regions if "auto"
    if language == "auto":
        language = detect_language_from_regions(detections, default="en")
        print(f"Auto-detected language: {'English' if language == 'en' else 'Nepali'}")

    for det in detections:
        if det.get("class") != "text_block_primary":
            continue

        x1, y1, x2, y2 = det["bbox"]
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        processed = preprocess(crop)

        # -----------------------
        # LANGUAGE ROUTING
        # -----------------------
        if language == "en":
            text, engine = _ocr_english(processed)

        elif language == "ne":
            text, engine = _ocr_nepali(processed)

        else:
            # Fallback: try English first, fallback Nepali
            text, engine = _ocr_english(processed)
            if not text:
                text, engine = _ocr_nepali(processed)

        if text:
            collected_text.append(text)
            engines_used.add(engine)

    full_text = " ".join(collected_text).strip()

    entities = []
    if full_text:
        entities = labeler.label_text(full_text)

    return {
        "text": full_text,
        "entities": entities,
        "ocr_engines_used": list(engines_used),
        "detected_language": language  # NEW: Return detected language
    }