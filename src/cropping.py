import os
from ultralytics import YOLO
import cv2

# Paths
image_dir = 'citizenship/images'
save_root_dir = 'citizenship/cropped_regions'
os.makedirs(save_root_dir, exist_ok=True)

# Load model
model = YOLO('runs/detect/train/weights/best.pt')  # update if needed

# Class names - adjust to your dataset classes
class_names = [
    "Id_card_boundary",
    "text_block_primary",
    "text_block_secondary",
    "fingerprint_region",
    "photo_region",
    "header_text_block"
]

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_name}")
        continue

    results = model(img)[0]

    base_name = os.path.splitext(img_name)[0]

    # Create subfolder for each original image inside cropped_regions
    save_dir = os.path.join(save_root_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)

    for i, (box, cls_id) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int))):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        class_name = class_names[cls_id]
        save_name = f"{base_name}-{class_name}_area_{i+1}.png"
        save_path = os.path.join(save_dir, save_name)

        cv2.imwrite(save_path, crop)
        print(f"Saved crop: {save_path}")

print("Cropping done.")
