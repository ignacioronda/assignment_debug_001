#!/usr/bin/env python3
"""
Automatically create YOLO labels by matching cropped building numbers 
to full validation images.
"""

import cv2
import numpy as np
from pathlib import Path

# Mapping of validation images to their building numbers
IMAGE_MAPPING = {
    'img1.jpg': None,  # NEGATIVE - "ENGINEERING" text
    'img2.jpg': '204',  # Has building number 204
    'img3.jpg': '314',  # Has building number 314
    'img4.jpg': '206',  # Has building number 206
    'img5.jpg': '215',  # Has building number 215
}

# You'll provide the cropped images here
CROP_DIR = Path('crops')  # Put your 4 cropped images here


def find_building_number_in_image(full_image_path, crop_image_path):
    """
    Find the location of the cropped building number in the full image.
    Returns (x1, y1, x2, y2) or None if not found.
    """
    full_img = cv2.imread(str(full_image_path))
    crop_img = cv2.imread(str(crop_image_path))
    
    if full_img is None or crop_img is None:
        return None
    
    # Convert to grayscale for template matching
    full_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    result = cv2.matchTemplate(full_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If match confidence is good
    if max_val > 0.6:
        x1, y1 = max_loc
        h, w = crop_gray.shape
        x2 = x1 + w
        y2 = y1 + h
        
        print(f"  Match found: confidence={max_val:.3f}, bbox=({x1},{y1},{x2},{y2})")
        return (x1, y1, x2, y2)
    
    print(f"  No good match found (best confidence: {max_val:.3f})")
    return None


def create_yolo_label(image_path, bbox):
    """
    Create YOLO format label file.
    """
    if bbox is None:
        print(f"  Skipping label creation (negative image)")
        return
    
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = bbox
    
    # Convert to YOLO format (normalized center x, y, width, height)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    
    # Create label file
    label_dir = Path('dataset/labels/val')
    label_dir.mkdir(parents=True, exist_ok=True)
    
    label_path = label_dir / f"{Path(image_path).stem}.txt"
    
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"  ✓ Created: {label_path}")


def save_verification_image(image_path, bbox):
    """
    Save image with bounding box for verification.
    """
    img = cv2.imread(str(image_path))
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, "Building Number", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save
    output_dir = Path('dataset/label_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(image_path).name
    cv2.imwrite(str(output_path), img)
    print(f"  Verification: {output_path}")


def main():
    val_image_dir = Path('dataset/images/val')
    crop_dir = CROP_DIR
    
    if not val_image_dir.exists():
        print(f"Error: {val_image_dir} not found")
        print("Please create dataset/images/val/ and put your validation images there")
        return
    
    if not crop_dir.exists():
        print(f"Error: {crop_dir} not found")
        print(f"Please create {crop_dir}/ and put your 4 cropped images there")
        print("Name them: 204.jpg, 206.jpg, 314.jpg, 215.jpg")
        return
    
    print("Creating labels from cropped images...\n")
    
    for img_name, building_num in IMAGE_MAPPING.items():
        print(f"Processing: {img_name}")
        
        image_path = val_image_dir / img_name
        
        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            continue
        
        if building_num is None:
            # Negative image
            print(f"  NEGATIVE - no label created")
            save_verification_image(image_path, None)
            continue
        
        # Find corresponding crop
        crop_path = crop_dir / f"{building_num}.jpg"
        if not crop_path.exists():
            crop_path = crop_dir / f"{building_num}.png"
        
        if not crop_path.exists():
            print(f"  ✗ Crop not found: {crop_path}")
            continue
        
        # Find building number in full image
        bbox = find_building_number_in_image(image_path, crop_path)
        
        if bbox:
            create_yolo_label(image_path, bbox)
            save_verification_image(image_path, bbox)
        else:
            print(f"  ✗ Could not locate building number in image")
    
    print("\n✓ Label creation complete!")
    print("Check dataset/label_verification/ to verify bounding boxes")


if __name__ == "__main__":
    main()