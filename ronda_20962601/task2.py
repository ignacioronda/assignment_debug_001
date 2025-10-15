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
import numpy as np
from pathlib import Path


def save_output(output_path, content, output_type='txt'):
    """Save output to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
    elif output_type == 'image':
        cv2.imwrite(output_path, content)


def select_best_characters(characters, max_chars=4):
    """
    Select 3-4 characters that are most likely to be digits.
    
    Strategy:
    1. Find characters with similar sizes (digits in same building number are similar)
    2. Pick the largest cluster of similar-sized characters
    3. Limit to 3-4 characters max
    
    Args:
        characters: List of character dicts with 'area', 'w', 'h', etc.
        max_chars: Maximum number of characters to return (default 4)
    
    Returns:
        Filtered list of character dicts
    """
    if len(characters) <= max_chars:
        return characters
    
    # Calculate normalized size (geometric mean of width and height)
    for char in characters:
        char['size'] = np.sqrt(char['w'] * char['h'])
    
    # Sort by size
    chars_by_size = sorted(characters, key=lambda c: c['size'], reverse=True)
    
    # Find clusters of similar-sized characters
    best_cluster = []
    best_cluster_score = 0
    
    # Try each character as a potential cluster center
    for i, center_char in enumerate(chars_by_size):
        cluster = [center_char]
        center_size = center_char['size']
        
        # Find other characters with similar size (within 50% range)
        for other_char in chars_by_size:
            if other_char == center_char:
                continue
            
            size_ratio = other_char['size'] / center_size
            
            # Similar size: within 0.5x to 2x range
            if 0.5 <= size_ratio <= 2.0:
                cluster.append(other_char)
        
        # Score this cluster: prefer larger characters and more characters
        avg_size = np.mean([c['size'] for c in cluster])
        cluster_score = len(cluster) * avg_size
        
        # Update best cluster if this is better
        if cluster_score > best_cluster_score and len(cluster) >= 2:
            best_cluster = cluster
            best_cluster_score = cluster_score
    
    # If no good cluster found, just take the largest characters
    if not best_cluster:
        best_cluster = chars_by_size[:max_chars]
    
    # Limit to max_chars
    if len(best_cluster) > max_chars:
        # Sort cluster by size and take largest
        best_cluster.sort(key=lambda c: c['size'], reverse=True)
        best_cluster = best_cluster[:max_chars]
    
    # Sort by x position (left to right)
    best_cluster.sort(key=lambda c: c['x'])
    
    return best_cluster


def segment_characters(img_path):
    """
    Segment individual characters from a building number image.
    
    Args:
        img_path: Path to building number image
    
    Returns:
        List of 3-4 character images, sorted left-to-right
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Extract ALL potential characters first
    characters = []
    
    for i in range(1, num_labels):
        x, y, w_c, h_c, area = stats[i]
        
        # **BOUNDARY CHECK** - ensure coordinates are within image bounds
        if x < 0 or y < 0:
            continue
        if x + w_c > w or y + h_c > h:
            # Clip to image boundaries
            w_c = min(w_c, w - x)
            h_c = min(h_c, h - y)
            if w_c <= 0 or h_c <= 0:
                continue
        
        # Very basic filters - be permissive at this stage
        if area < 50:  # Skip tiny noise
            continue
        
        if w_c < 5 or h_c < 10:  # Skip very thin/short
            continue
        
        if w_c > w * 0.95 or h_c > h * 0.95:  # Skip whole image
            continue
        
        # Skip extremely wide things (horizontal lines, etc.)
        if h_c > 0:
            aspect = w_c / h_c
            if aspect > 3.5:  # Very wide
                continue
            if aspect < 0.15:  # Very thin vertical line
                continue
        
        # Extract character - with boundary safety
        try:
            char_img = gray[y:y+h_c, x:x+w_c]
            
            # Verify extraction succeeded
            if char_img.size == 0:
                continue
                
        except Exception as e:
            print(f"    Warning: Failed to extract component at ({x},{y},{w_c},{h_c}): {e}")
            continue
        
        characters.append({
            'image': char_img,
            'x': x,
            'y': y,
            'w': w_c,
            'h': h_c,
            'area': area
        })
    
    if not characters:
        return []
    
    # Apply smart selection to pick 3-4 best characters
    best_characters = select_best_characters(characters, max_chars=4)
    
    # Final sort by x position
    best_characters.sort(key=lambda c: c['x'])
    
    return best_characters


def run_task2(image_path, config):
    """
    Task 2: Character Segmentation
    
    Reads building number images from input directory and segments
    individual characters, saving them as c1.png, c2.png, etc.
    """
    print(f"\n{'='*60}")
    print(f"TASK 2: Character Segmentation")
    print(f"{'='*60}")
    
    input_path = Path(image_path)
    
    # Handle both single file and directory
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(list(input_path.glob('bn*.png')) + list(input_path.glob('bn*.jpg')))
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