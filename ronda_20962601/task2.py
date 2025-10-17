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

# (Copy full license header from your current file)
# Author: Ignacio Ronda, Student ID: 20962601

"""
Task 2: Character Segmentation using Connected Components

Segments individual digits from building number images using classical CV.
I chose connected components over deep learning because it's simpler and
doesn't require training data.

Key challenge: Filtering out brick texture and noise while keeping real digits.
My solution: Cluster by size - real digits are usually similar-sized!

Performance: 12 characters segmented from 4 building numbers
"""

import os
import cv2
import numpy as np
from pathlib import Path

def save_output(output_path, content, output_type='txt'):
    """Save output - either text or image"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
    elif output_type == 'image':
        cv2.imwrite(output_path, content)

def select_best_characters(characters, max_chars=4):
    """
    Smart character selection - picks 3-4 most likely digits.
    
    Strategy:
    1. Calculate size for each potential character
    2. Find clusters of similar-sized characters
    3. Pick the largest cluster (these are likely the real digits)
    4. Return top 3-4 characters, sorted left-to-right
    
    Why this works: Real digits in same building number have similar sizes,
    but noise/shadows/brick texture have very different sizes.
    """
    # If we already have 4 or fewer, just return them all
    if len(characters) <= max_chars:
        return characters
    
    # Calculate normalized size (geometric mean of width and height)
    for char in characters:
        char['size'] = np.sqrt(char['w'] * char['h'])
    
    # Sort by size (largest first)
    chars_by_size = sorted(characters, key=lambda c: c['size'], reverse=True)
    
    # Find best cluster of similar-sized characters
    best_cluster = []
    best_cluster_score = 0
    
    # Try each character as a potential cluster center
    for center_char in chars_by_size:
        cluster = [center_char]
        center_size = center_char['size']
        
        # Find other characters with similar size
        for other_char in chars_by_size:
            if other_char == center_char:
                continue
            
            size_ratio = other_char['size'] / center_size
            
            # Similar size: within 0.5x to 2x range
            if 0.5 <= size_ratio <= 2.0:
                cluster.append(other_char)
        
        # Score this cluster (bigger clusters with bigger characters are better)
        avg_size = np.mean([c['size'] for c in cluster])
        cluster_score = len(cluster) * avg_size
        
        # Update best if this is better
        if cluster_score > best_cluster_score and len(cluster) >= 2:
            best_cluster = cluster
            best_cluster_score = cluster_score
    
    # If no good cluster, just take largest characters
    if not best_cluster:
        best_cluster = chars_by_size[:max_chars]
    
    # Limit to max_chars
    if len(best_cluster) > max_chars:
        best_cluster.sort(key=lambda c: c['size'], reverse=True)
        best_cluster = best_cluster[:max_chars]
    
    # Sort by x position (left to right) for correct digit order
    best_cluster.sort(key=lambda c: c['x'])
    
    return best_cluster

def segment_characters(img_path):
    """
    Segment individual characters from building number image.
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur (reduce noise)
    3. Otsu's thresholding (automatic binarization)
    4. Find connected components (potential characters)
    5. Filter by size and aspect ratio
    6. Select best 3-4 characters using clustering
    
    Returns: List of character images
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    # Convert to grayscale (easier to process)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding - automatically finds best threshold
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert (we want white text on black)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # Morphological closing - connects broken character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find connected components (blobs of white pixels)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Extract potential characters
    characters = []
    
    # Loop through each component (skip 0 which is background)
    for i in range(1, num_labels):
        x, y, w_c, h_c, area = stats[i]
        
        # BOUNDARY CHECK - make sure coordinates are valid
        if x < 0 or y < 0:
            continue
        if x + w_c > w or y + h_c > h:
            # Clip to image boundaries
            w_c = min(w_c, w - x)
            h_c = min(h_c, h - y)
            if w_c <= 0 or h_c <= 0:
                continue
        
        # Filter 1: Skip tiny blobs (noise)
        if area < 50:
            continue
        
        # Filter 2: Skip very thin/short things
        if w_c < 5 or h_c < 10:
            continue
        
        # Filter 3: Skip huge things (probably the whole image)
        if w_c > w * 0.95 or h_c > h * 0.95:
            continue
        
        # Filter 4: Skip weird aspect ratios
        if h_c > 0:
            aspect = w_c / h_c
            if aspect > 3.5:  # Too wide (horizontal line)
                continue
            if aspect < 0.15:  # Too thin (vertical line)
                continue
        
        # Extract character image
        try:
            char_img = gray[y:y+h_c, x:x+w_c]
            
            # Make sure extraction worked
            if char_img.size == 0:
                continue
                
        except Exception as e:
            # If something goes wrong, skip this component
            continue
        
        # Store character info
        characters.append({
            'image': char_img,
            'x': x,
            'y': y,
            'w': w_c,
            'h': h_c,
            'area': area
        })
    
    # If no characters found, return empty
    if not characters:
        return []
    
    # Apply smart selection to pick 3-4 best characters
    best_characters = select_best_characters(characters, max_chars=4)
    
    # Sort by x position (left to right)
    best_characters.sort(key=lambda c: c['x'])
    
    return best_characters

def run_task2(image_path, config):
    """Main function for Task 2"""
    print(f"\n{'='*60}")
    print(f"TASK 2: Character Segmentation")
    print(f"{'='*60}")
    
    input_path = Path(image_path)
    
    # Handle both single file and directory
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(list(input_path.glob('bn*.png')) + 
                           list(input_path.glob('bn*.jpg')))
    else:
        print(f"ERROR: {image_path} is not a valid file or directory")
        return
    
    if not image_files:
        print(f"No building number images found in: {image_path}")
        return
    
    print(f"Processing {len(image_files)} building number(s)...\n")
    
    total_segmented = 0
    results = {}
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        try:
            # Segment characters
            characters = segment_characters(img_file)
            
            if not characters:
                print(f"  ✗ No characters segmented")
                results[img_file.name] = 0
                continue
            
            print(f"  ✓ Segmented {len(characters)} character(s)")
            results[img_file.name] = len(characters)
            
            # Create output directory
            bn_name = img_file.stem
            output_dir = f"output/task2/{bn_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each character
            for idx, char_data in enumerate(characters, start=1):
                char_img = char_data['image']
                
                output_filename = f"c{idx}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                save_output(output_path, char_img, output_type='image')
                size = np.sqrt(char_data['w'] * char_data['h'])
                print(f"    Saved: {output_filename} ({char_data['w']}x{char_data['h']}, size={size:.1f})")
            
            total_segmented += len(characters)
            
        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {e}")
            results[img_file.name] = 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Results per image:")
    for img_name, count in results.items():
        status = "✓" if count >= 2 else "✗"
        print(f"    {status} {img_name}: {count} character(s)")
    print(f"\n  Total characters segmented: {total_segmented}")
    print(f"  Pass requirement: 6 characters")
    print(f"  Status: {'✓ PASS' if total_segmented >= 6 else '✗ FAIL'}")
    print(f"{'='*60}\n")