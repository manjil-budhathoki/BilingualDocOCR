import os
import cv2
import json
import re
from language_detector import get_language_from_folder  # NEW IMPORT

# ===============================
# CONFIG
# ===============================

CROPS_ROOT_DIR = "citizenship/cropped_regions"
OUTPUT_PATH = "Result/test_results.json"


# ===============================
# SHARED UTILS (SAFE TO IMPORT)
# ===============================

def get_base(folder):
    return re.sub(r'_(front|back|side)$', '', folder, re.IGNORECASE)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


# REMOVED: Old get_language function - now using imported one


def build_engine_info(lang, engine):
    return {
        "language": "english" if lang == "en" else "nepali",
        "ocr_engine": engine,
        "model_family": "DocTR" if engine == "doctr" else "EasyOCR"
    }


# ===============================
# BATCH OCR (SCRIPT MODE ONLY)
# ===============================

def run_batch_ocr(max_folders=40):
    output = {}

    folders = [
        f for f in os.listdir(CROPS_ROOT_DIR)
        if os.path.isdir(os.path.join(CROPS_ROOT_DIR, f))
    ][:max_folders]

    for folder in folders:
        path = os.path.join(CROPS_ROOT_DIR, folder)
        base = get_base(folder)

        # NEW: Use unified language detector
        lang = get_language_from_folder(path, default="ne")
        print(f"\n{base}: {'English' if lang == 'en' else 'Nepali'} (detected from regions)")

        output[base] = []

        for img_name in os.listdir(path):
            if "primary" not in img_name.lower():
                continue
            if not img_name.lower().endswith(".png"):
                continue

            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                output[base].append({
                    "file": img_name,
                    "text": "[Error loading image]",
                    "engine": None
                })
                continue

            processed = preprocess(image)

            # ==========================
            # ENGLISH → DocTR + fallback
            # ==========================
            if lang == "en":
                try:
                    from doctr.io import DocumentFile
                    from doctr.models import ocr_predictor
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as tmp:
                        cv2.imwrite(tmp.name, processed)

                        model = ocr_predictor(pretrained=True)
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
                    engine = "doctr"

                except Exception:
                    import easyocr
                    reader = easyocr.Reader(["en"], gpu=False)
                    result = reader.readtext(processed, detail=0)
                    text = " ".join(result).strip()
                    engine = "easyocr_fallback"

            # ==========================
            # NEPALI → EasyOCR
            # ==========================
            else:
                import easyocr
                reader = easyocr.Reader(["ne", "en"], gpu=False)
                result = reader.readtext(processed, detail=0)
                text = " ".join(result).strip()
                engine = "easyocr"

            output[base].append({
                "file": img_name,
                "text": text if text else "[No text]",
                "engine": build_engine_info(lang, engine)
            })

            print(f"  {img_name}: {text[:60]}...")

    # ===============================
    # DOCUMENT-LEVEL SUMMARY
    # ===============================

    summary = {}
    for doc, pages in output.items():
        engines_used = {
            p["engine"]["ocr_engine"]
            for p in pages
            if p.get("engine")
        }
        summary[doc] = {
            "engines_used": list(engines_used),
            "primary_engine": list(engines_used)[0] if engines_used else None
        }

    final_output = {
        "documents": output,
        "summary": summary
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved OCR results to {OUTPUT_PATH}")


# ===============================
# ENTRY POINT (CRITICAL)
# ===============================

if __name__ == "__main__":
    run_batch_ocr()