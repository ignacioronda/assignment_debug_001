#!/usr/bin/env python3
"""
Manual labeling helper for building numbers.

Simple click-and-drag interface to label building numbers in images.
Useful for images where automatic template matching failed.

Usage:
    python manual_label_helper.py
    
Instructions:
    - Left click and drag to draw bounding box
    - Press 's' to save and move to next image
    - Press 'r' to reset current box
    - Press 'q' to quit
"""

import cv2
import numpy as np
from pathlib import Path

# Configuration
IMAGE_DIR = Path('validation/task1')
OUTPUT_LABEL_DIR = Path('dataset_v2/labels/manual')

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
current_box = None


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback function for drawing bounding box."""
    global ix, iy, drawing, current_box, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_display.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Label Building Numbers', img_temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        img_temp = img_display.copy()
        cv2.rectangle(img_temp, (current_box[0], current_box[1]), 
                     (current_box[2], current_box[3]), (0, 255, 0), 2)
        cv2.imshow('Label Building Numbers', img_temp)


def save_yolo_label(img_path, bbox, label_dir):
    """Save bounding box in YOLO format."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = bbox
    
    # Convert to YOLO format (normalized center x, y, width, height)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    
    # Save label
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{img_path.stem}.txt"
    
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✓ Saved label: {label_path.name}")


def label_images():
    """Main labeling interface."""
    global img_display, current_box
    
    print("="*60)
    print("Manual Labeling Helper")
    print("="*60)
    print("\nInstructions:")
    print("  - Left click and drag to draw bounding box")
    print("  - Press 's' to save and move to next image")
    print("  - Press 'r' to reset current box")
    print("  - Press 'n' to skip image (no building number)")
    print("  - Press 'q' to quit")
    print("="*60)
    
    # Get all images
    image_files = sorted(list(IMAGE_DIR.glob('*.jpg')) + list(IMAGE_DIR.glob('*.png')))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Create window
    cv2.namedWindow('Label Building Numbers')
    cv2.setMouseCallback('Label Building Numbers', draw_rectangle)
    
    for idx, img_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        img_display = img.copy()
        current_box = None
        
        # Display
        cv2.imshow('Label Building Numbers', img_display)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Save and next
            if key == ord('s'):
                if current_box is not None:
                    save_yolo_label(img_path, current_box, OUTPUT_LABEL_DIR)
                    break
                else:
                    print("⚠ No box drawn! Draw a box first or press 'n' to skip.")
            
            # Reset box
            elif key == ord('r'):
                current_box = None
                img_display = img.copy()
                cv2.imshow('Label Building Numbers', img_display)
                print("  Reset box")
            
            # Skip (negative image)
            elif key == ord('n'):
                print("  Skipped (negative image)")
                break
            
            # Quit
            elif key == ord('q'):
                print("\nQuitting...")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("Labeling complete!")
    print(f"Labels saved to: {OUTPUT_LABEL_DIR}")
    print("="*60)


if __name__ == "__main__":
    label_images()