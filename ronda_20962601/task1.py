# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Ignacio Ronda
# Student ID: 20962601
# Last Modified: 2025-10-16

"""
Task 1: Building Number Detection using YOLOv8

This program detects building numbers in campus photos using a custom-trained
YOLOv8 object detection model. I trained it on 78 augmented images to improve
performance from the baseline 50% to 81% detection rate.

Approach:
- YOLOv8 nano model (fast and lightweight)
- Very low confidence threshold (0.01) to catch everything
- Trained on augmented dataset with various lighting/angles
- 15-pixel padding around detections for better downstream processing

Performance achieved:
- 81% detection rate (13/16 images)
- Successfully handles both Type I (black plates) and Type II (wall-mounted)
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO  # YOLOv8 library for object detection


def save_output(output_path, content, output_type='txt'):
    """
    Helper function to save outputs (either text files or images).
    
    Args:
        output_path: Where to save the file
        content: What to save (string for text, numpy array for images)
        output_type: 'txt' or 'image'
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        # Write text content to file
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Save image using OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")


def run_task1(image_path, config):
    """
    Main function for Task 1: Building Number Detection
    
    This function:
    1. Loads the trained YOLOv8 model
    2. Processes all images in the input directory
    3. Detects building numbers using object detection
    4. Saves detected regions as bnX.png files
    5. Skips negative images (no detections)
    
    Args:
        image_path: Path to directory containing images to process
        config: Configuration dictionary (not used, but required by framework)
    """
    
    # Print header for clarity
    print(f"\n{'='*60}")
    print(f"TASK 1: Building Number Detection (YOLOv8)")
    print(f"{'='*60}")
    
    # Path to my trained model
    # This model was trained for 75 epochs on 78 augmented samples
    model_path = 'runs/building_numbers_v2/train/weights/best.pt'
    
    # Check if model exists before trying to load it
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python train_yolo_v2.py")
        return
    
    # Load the trained YOLOv8 model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Convert string path to Path object for easier file handling
    input_path = Path(image_path)
    
    # Handle both single file and directory inputs
    if input_path.is_file():
        # If it's a single file, process just that file
        image_files = [input_path]
    elif input_path.is_dir():
        # If it's a directory, get all jpg and png files
        # sorted() ensures consistent ordering
        image_files = sorted(list(input_path.glob('*.jpg')) + 
                           list(input_path.glob('*.png')))
    else:
        print(f"ERROR: {image_path} is not a valid file or directory")
        return
    
    # Check if we found any images
    if not image_files:
        print(f"No images found in: {image_path}")
        return
    
    print(f"Processing {len(image_files)} image(s)...\n")
    
    # Track statistics for summary
    positive_count = 0  # Images with building numbers detected
    negative_count = 0  # Images with no detections
    
    # Process each image one by one
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        # Load image using OpenCV (reads as BGR format)
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ✗ Failed to load")
            continue
        
        # Print image dimensions for debugging
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
        
        # Run YOLOv8 inference
        # conf=0.01: Very low confidence threshold to catch everything
        # verbose=False: Don't print detailed detection info
        results = model(image, conf=0.01, verbose=False)
        
        # Extract all detections from results
        detections = []
        for result in results:
            boxes = result.boxes  # Get bounding boxes from result
            for box in boxes:
                # Extract box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Extract confidence score
                conf = box.conf[0].cpu().numpy()
                
                # Store detection info
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'confidence': float(conf)
                })
        
        print(f"  Raw detections: {len(detections)}")
        
        # If no detections at all, this is a negative image
        if not detections:
            print(f"  ✗ NEGATIVE - No detections at all")
            negative_count += 1
            continue
        
        # Take the detection with HIGHEST confidence
        # This handles cases where multiple things are detected
        best = max(detections, key=lambda d: d['confidence'])
        print(f"  Best detection confidence: {best['confidence']:.3f}")
        
        # Filter out very low confidence detections (likely noise)
        # 0.02 threshold was determined through experimentation
        if best['confidence'] < 0.02:
            print(f"  ✗ NEGATIVE - Confidence too low (likely noise)")
            negative_count += 1
            continue
        
        # If we get here, we have a valid detection!
        print(f"  ✓ POSITIVE")
        positive_count += 1
        
        # Extract the detected region from the image
        x, y, w, h = best['bbox']
        
        # Add padding around detection for better downstream processing
        # Padding helps Task 2 (segmentation) work better
        pad = 15
        x1 = max(0, x - pad)  # Don't go negative
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)  # Don't exceed image width
        y2 = min(image.shape[0], y + h + pad)  # Don't exceed image height
        
        # Crop the building number region
        extracted = image[y1:y2, x1:x2]
        
        # Generate output filename
        img_name = img_file.stem  # Get filename without extension
        
        # Handle different naming conventions (img1, image1, etc.)
        if img_name.startswith('img'):
            img_idx = img_name[3:]  # "img1" -> "1"
        elif img_name.startswith('image'):
            img_idx = img_name[5:]  # "image1" -> "1"
        else:
            img_idx = img_name  # Use as-is
        
        # Output filename: bn + number + .png
        output_filename = f"bn{img_idx}.png"
        output_path = f"output/task1/{output_filename}"
        
        # Save the extracted region
        save_output(output_path, extracted, output_type='image')
        print(f"  ✓ Saved: {output_filename}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Positive (saved): {positive_count}")
    print(f"  Negative (skipped): {negative_count}")
    print(f"  Total processed: {len(image_files)}")
    print(f"{'='*60}\n")