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
# Last Modified: 2025-10-04

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def save_output(output_path, content, output_type='txt'):
    """Save output to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")


def run_task1(image_path, config):
    """
    Task 1: Building Number Detection using Custom YOLOv8
    
    Uses a custom-trained YOLOv8 model specifically for building numbers.
    Processes any number of images in a directory or a single image file.
    """
    print(f"\n{'='*60}")
    print(f"TASK 1: Building Number Detection (YOLOv8)")
    print(f"{'='*60}")
    
    # Load trained model
    model_path = 'runs/building_numbers/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python train_yolo.py")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    input_path = Path(image_path)
    
    # Handle both single file and directory
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        # Get all jpg and png files
        image_files = sorted(list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')))
    else:
        print(f"ERROR: {image_path} is not a valid file or directory")
        return
    
    if not image_files:
        print(f"No images found in: {image_path}")
        return
    
    print(f"Processing {len(image_files)} image(s)...\n")
    
    positive_count = 0
    negative_count = 0
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ✗ Failed to load")
            continue
        
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
        
        # Run inference - use VERY low confidence to catch everything
        results = model(image, conf=0.01, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'confidence': float(conf)
                })
        
        print(f"  Raw detections: {len(detections)}")
        
        if not detections:
            print(f"  ✗ NEGATIVE - No detections at all")
            negative_count += 1
            continue
        
        # Take the HIGHEST confidence detection (even if low)
        best = max(detections, key=lambda d: d['confidence'])
        print(f"  Best detection confidence: {best['confidence']:.3f}")
        
        # Very lenient threshold - accept anything above random noise
        if best['confidence'] < 0.02:
            print(f"  ✗ NEGATIVE - Confidence too low (likely noise)")
            negative_count += 1
            continue
        
        print(f"  ✓ POSITIVE")
        positive_count += 1
        
        # Extract region with padding
        x, y, w, h = best['bbox']
        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        extracted = image[y1:y2, x1:x2]
        
        # Generate output filename
        img_name = img_file.stem
        
        # Handle different naming conventions
        if img_name.startswith('img'):
            img_idx = img_name[3:]  # img1 -> 1
        elif img_name.startswith('image'):
            img_idx = img_name[5:]  # image1 -> 1
        else:
            img_idx = img_name  # Use as-is
        
        output_filename = f"bn{img_idx}.png"
        output_path = f"output/task1/{output_filename}"
        
        save_output(output_path, extracted, output_type='image')
        print(f"  ✓ Saved: {output_filename}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Positive (saved): {positive_count}")
    print(f"  Negative (skipped): {negative_count}")
    print(f"  Total processed: {len(image_files)}")
    print(f"{'='*60}\n")