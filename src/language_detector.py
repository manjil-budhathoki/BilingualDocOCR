"""
language_detector.py
Unified language detection based on YOLO region detections
"""

def detect_language_from_regions(detections, default="ne"):
    """
    Determine document language based on detected regions.
    
    Logic:
    - If fingerprint_region detected → English (back side)
    - If photo_region detected → Nepali (front side)
    - If both or neither → use priority or default
    
    Args:
        detections: List of dicts with 'class' field from YOLO
        default: Default language if ambiguous ("en" or "ne")
    
    Returns:
        "en" for English, "ne" for Nepali
    """
    fingerprint_found = False
    photo_found = False
    
    # Scan all detections
    for det in detections:
        if isinstance(det, dict) and 'class' in det:
            class_name = det['class']
            if class_name == "fingerprint_region":
                fingerprint_found = True
            elif class_name == "photo_region":
                photo_found = True
    
    # Decision logic
    if fingerprint_found and not photo_found:
        return "en"  # English document (back side with fingerprint)
    elif photo_found and not fingerprint_found:
        return "ne"  # Nepali document (front side with photo)
    elif fingerprint_found and photo_found:
        # Both detected - use priority or analyze more
        # You might want to check which has higher confidence or area
        return "ne"  # Default to Nepali for mixed documents
    else:
        # Neither detected - use default
        return default


def get_language_from_folder(folder_path, default="ne"):
    """
    Alternative for batch processing: detect based on filenames in folder
    (Maintains compatibility with existing batch code)
    """
    import os
    
    fingerprint_files = []
    photo_files = []
    
    for filename in os.listdir(folder_path):
        if "fingerprint" in filename.lower():
            fingerprint_files.append(filename)
        if "photo" in filename.lower():
            photo_files.append(filename)
    
    # Same logic as above but with filenames
    if fingerprint_files and not photo_files:
        return "en"
    elif photo_files and not fingerprint_files:
        return "ne"
    elif fingerprint_files and photo_files:
        return default
    else:
        return default