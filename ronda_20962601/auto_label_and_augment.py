#!/usr/bin/env python3
"""
Automatic labeling and augmentation script for YOLOv8 training.

This script:
1. AUTO-DISCOVERS all bnX crops and matches to imgX images
2. Generates YOLO format labels automatically
3. Applies data augmentation to expand the dataset
4. Prepares everything for YOLOv8 training

Usage:
    python auto_label_and_augment.py
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
CROP_DIR = Path('validation/task2')  # Where your cropped images are
FULL_DIR = Path('validation/task1')  # Where your full images are
OUTPUT_DIR = Path('dataset_v2')      # New dataset directory

# Augmentation settings
AUGMENT_ENABLED = True
AUGMENTATIONS_PER_IMAGE = 5  # How many augmented versions per image

# YOLOv8 settings
TRAIN_SPLIT = 0.8  # 80% training, 20% validation


# ============================================================================
# AUTO-DISCOVERY: Automatically find all bnX -> imgX mappings
# ============================================================================

def discover_crop_mappings(crop_dir, full_dir):
    """
    Automatically discover all bn* crops and map to img* images.
    
    Args:
        crop_dir: Path to directory containing bnX.png files (NOT subdirectories)
        full_dir: Path to directory containing imgX.jpg files
    
    Returns:
        dict: {crop_name: crop_file_path, image_name: image_file_path} tuples
    """
    mappings = {}
    
    # Find all crop FILES (bnX.png, bnX.jpg, etc.)
    if not crop_dir.exists():
        print(f"ERROR: Crop directory not found: {crop_dir}")
        return mappings
    
    # Look for crop files directly in the directory
    crop_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
        crop_files.extend(crop_dir.glob(f"bn*{ext}"))
    
    for crop_file_path in crop_files:
        crop_name = crop_file_path.stem  # e.g., "bn2", "bn3", "bn11"
        
        if not crop_name.startswith('bn'):
            continue
        
        # Extract number from bnX
        number = crop_name[2:]  # Remove "bn" prefix
        
        # Map to imgX
        img_name = f"img{number}"
        
        # Check if corresponding image exists
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            potential_img = full_dir / f"{img_name}{ext}"
            if potential_img.exists():
                img_file = potential_img
                break
        
        if img_file:
            mappings[crop_name] = {
                'crop_file': crop_file_path,
                'img_file': img_file,
                'img_name': img_name
            }
        else:
            print(f"  ⚠ Warning: Found {crop_name} but no matching {img_name}")
    
    return mappings


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_crop_in_image(full_img_path, crop_img_path, threshold=0.6):
    """
    Find the location of a cropped region in a full image using template matching.
    
    Args:
        full_img_path: Path to full image
        crop_img_path: Path to cropped region
        threshold: Matching confidence threshold (0-1)
    
    Returns:
        (x1, y1, x2, y2) bounding box or None if not found
    """
    full_img = cv2.imread(str(full_img_path))
    crop_img = cv2.imread(str(crop_img_path))
    
    if full_img is None or crop_img is None:
        return None
    
    # Convert to grayscale for better matching
    full_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    result = cv2.matchTemplate(full_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If match is good enough
    if max_val >= threshold:
        x1, y1 = max_loc
        h, w = crop_gray.shape
        x2 = x1 + w
        y2 = y1 + h
        
        return (x1, y1, x2, y2)
    
    return None


def bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixels
        img_width: Image width
        img_height: Image height
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)


def augment_image(img, bbox):
    """
    Apply random augmentation to image and adjust bounding box.
    
    Args:
        img: Input image (numpy array)
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Augmented image and adjusted bounding box
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness = random.uniform(0.7, 1.3)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    if random.random() > 0.5:
        contrast = random.uniform(0.8, 1.2)
        img = np.clip((img - 128) * contrast + 128, 0, 255).astype(np.uint8)
    
    # Random rotation (-15 to +15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        
        # Rotate bounding box corners
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        rotated_corners = M @ corners
        
        # Get new bounding box
        x1 = int(np.min(rotated_corners[0]))
        y1 = int(np.min(rotated_corners[1]))
        x2 = int(np.max(rotated_corners[0]))
        y2 = int(np.max(rotated_corners[1]))
        
        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
    
    # Random horizontal flip (less common for building numbers)
    if random.random() > 0.8:
        img = cv2.flip(img, 1)
        x1_new = w - x2
        x2_new = w - x1
        x1, x2 = x1_new, x2_new
    
    # Random Gaussian blur
    if random.random() > 0.7:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Random noise
    if random.random() > 0.8:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    return img, (x1, y1, x2, y2)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_dataset():
    """
    Main function to create augmented dataset with YOLO labels.
    """
    
    print("="*60)
    print("Building Number Detection - Dataset Creation")
    print("="*60)
    
    # Create output directories
    output_dirs = [
        OUTPUT_DIR / 'images' / 'train',
        OUTPUT_DIR / 'images' / 'val',
        OUTPUT_DIR / 'labels' / 'train',
        OUTPUT_DIR / 'labels' / 'val',
    ]
    
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created output directories in: {OUTPUT_DIR}")
    
    # Auto-discover crop mappings
    print("\n" + "="*60)
    print("Step 0: Auto-discovering crop to image mappings")
    print("="*60)
    
    crop_mappings = discover_crop_mappings(CROP_DIR, FULL_DIR)
    
    if not crop_mappings:
        print("\n✗ No valid crop-to-image mappings found!")
        print(f"  Make sure you have:")
        print(f"  - Crops in: {CROP_DIR}/bnX.png")
        print(f"  - Images in: {FULL_DIR}/imgX.jpg")
        return
    
    print(f"\n✓ Found {len(crop_mappings)} building numbers:")
    for crop_name, mapping_info in sorted(crop_mappings.items()):
        print(f"  {crop_name} → {mapping_info['img_name']}")
    
    # Process each cropped image
    all_samples = []
    
    print("\n" + "="*60)
    print("Step 1: Matching crops to full images")
    print("="*60)
    
    for crop_name, mapping_info in crop_mappings.items():
        crop_file = mapping_info['crop_file']
        full_file = mapping_info['img_file']
        img_name = mapping_info['img_name']
        
        print(f"\nProcessing: {crop_name} -> {img_name}")
        
        # Find bounding box
        bbox = find_crop_in_image(full_file, crop_file)
        
        if bbox is None:
            print(f"  ✗ Could not find crop in image (template matching failed)")
            continue
        
        print(f"  ✓ Found at: {bbox}")
        
        # Load image
        img = cv2.imread(str(full_file))
        
        # Store original
        all_samples.append({
            'image': img.copy(),
            'bbox': bbox,
            'name': f"{img_name}_orig"
        })
        
        # Generate augmented versions
        if AUGMENT_ENABLED:
            print(f"  Generating {AUGMENTATIONS_PER_IMAGE} augmented versions...")
            for i in range(AUGMENTATIONS_PER_IMAGE):
                aug_img, aug_bbox = augment_image(img.copy(), bbox)
                all_samples.append({
                    'image': aug_img,
                    'bbox': aug_bbox,
                    'name': f"{img_name}_aug{i+1}"
                })
    
    print(f"\n✓ Total samples generated: {len(all_samples)}")
    
    # Split into train/val
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_SPLIT)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    
    # Save samples
    print("\n" + "="*60)
    print("Step 2: Saving images and labels")
    print("="*60)
    
    for split_name, samples in [('train', train_samples), ('val', val_samples)]:
        for idx, sample in enumerate(samples):
            img = sample['image']
            bbox = sample['bbox']
            name = sample['name']
            
            h, w = img.shape[:2]
            
            # Save image
            img_filename = f"{name}_{idx:04d}.jpg"
            img_path = OUTPUT_DIR / 'images' / split_name / img_filename
            cv2.imwrite(str(img_path), img)
            
            # Convert bbox to YOLO format
            yolo_bbox = bbox_to_yolo(bbox, w, h)
            
            # Save label (class_id x_center y_center width height)
            label_filename = f"{name}_{idx:04d}.txt"
            label_path = OUTPUT_DIR / 'labels' / split_name / label_filename
            
            with open(label_path, 'w') as f:
                f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    print(f"\n✓ Saved all images and labels")
    
    # Create data.yaml
    print("\n" + "="*60)
    print("Step 3: Creating data.yaml")
    print("="*60)
    
    data_yaml_content = f"""# Building Number Detection Dataset
path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['building_number']
"""
    
    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✓ Created: {yaml_path}")
    
    print("\n" + "="*60)
    print("Dataset Creation Complete!")
    print("="*60)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"Total samples: {len(all_samples)}")
    print(f"  - Training: {len(train_samples)}")
    print(f"  - Validation: {len(val_samples)}")
    print(f"\nNext steps:")
    print(f"  1. Review the dataset (check images and labels)")
    print(f"  2. Train YOLOv8: python train_yolo_v2.py")
    print("="*60)


if __name__ == "__main__":
    create_dataset()