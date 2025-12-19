import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from language_detector import detect_language_from_regions  # NEW IMPORT

from NER.ocr_ner_pipeline import process_image

st.set_page_config(layout="wide", page_title="Nepali OCR + NER")
st.title("📄 Nepali Document OCR & NER")

@st.cache_resource
def load_yolo():
    return YOLO("runs/detect/train/weights/best.pt")  # or your trained model

yolo = load_yolo()

CLASS_NAMES = [
    "Id_card_boundary",
    "text_block_primary",
    "text_block_secondary",
    "fingerprint_region",
    "photo_region",
    "header_text_block",
]

# NEW: Language selector with auto-detect option
language_option = st.selectbox(
    "OCR Language",
    options=["auto", "en", "ne"],
    format_func=lambda x: {
        "auto": "Auto-detect from document",
        "en": "English (force)",
        "ne": "Nepali (force)"
    }[x]
)

uploaded = st.file_uploader("Upload document image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = cv2.imdecode(
        np.frombuffer(uploaded.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    with st.spinner("Running detection..."):
        result = yolo(image)[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.cpu().numpy()
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            detections.append({
                "bbox": list(map(int, box)),
                "class": CLASS_NAMES[int(cls)],
                "confidence": float(conf)
            })
    
    # NEW: Show detected regions summary
    region_counts = {}
    for d in detections:
        cls = d["class"]
        region_counts[cls] = region_counts.get(cls, 0) + 1
    
    if region_counts:
        st.sidebar.subheader("📊 Detected Regions")
        for region, count in region_counts.items():
            st.sidebar.write(f"• {region}: {count}")

    with st.spinner("Running OCR + NER..."):
        output = process_image(image, detections, language=language_option)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Regions")
        annotated = image.copy()
        for d in detections:
            if d["class"] == "text_block_primary":
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optional: Color code different regions
            elif d["class"] == "fingerprint_region":
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif d["class"] == "photo_region":
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        # NEW: Show detected language
        detected_lang = output.get("detected_language", "unknown")
        lang_display = {
            "en": "🇬🇧 English",
            "ne": "🇳🇵 Nepali",
            "unknown": "Unknown"
        }.get(detected_lang, detected_lang)
        
        st.subheader(f"OCR Results ({lang_display})")
        st.text_area("Extracted text", output["text"], height=220)

        st.subheader("Entities")
        if output["entities"]:
            for e in output["entities"]:
                st.write(f"**{e.label}** → {e.text}")
        else:
            st.info("No entities found")
        
        # NEW: Show engines used
        if output.get("ocr_engines_used"):
            st.caption(f"Engines used: {', '.join(output['ocr_engines_used'])}")