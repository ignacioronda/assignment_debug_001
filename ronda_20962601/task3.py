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

# Force CPU usage (avoid CUDA issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


def save_output(output_path, content, output_type='txt'):
    """Save output to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
    elif output_type == 'image':
        cv2.imwrite(output_path, content)


def preprocess_character_image(img_path, target_size=(28, 28)):
    """
    Preprocess a character image for CNN prediction.
    
    Args:
        img_path: Path to character image
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array ready for CNN input
    """
    # Read image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Resize to 28x28 (MNIST size)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Check if image is white text on black background or vice versa
    # MNIST is white digits on black background
    mean_val = np.mean(img_resized)
    
    if mean_val > 127:  # Black text on white background
        img_resized = 255 - img_resized  # Invert
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1) for CNN input
    img_input = img_normalized.reshape(1, 28, 28, 1)
    
    return img_input


def load_digit_classifier(model_path='data/digit_classifier.h5'):
    """
    Load the trained CNN digit classifier.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        # Try alternative path
        alt_path = 'data/digit_classifier.keras'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train the model first using: python train_digit_classifier.py"
            )
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    return model


def recognize_character(img_path, model):
    """
    Recognize a single character using the CNN model.
    
    Args:
        img_path: Path to character image
        model: Trained Keras model
    
    Returns:
        Tuple of (predicted_digit, confidence)
    """
    # Preprocess image
    img_input = preprocess_character_image(img_path)
    
    if img_input is None:
        return None, 0.0
    
    # Predict
    predictions = model.predict(img_input, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence


def run_task3(image_path, config):
    """
    Task 3: Character Recognition
    
    Reads individual character images and recognizes them using
    a trained CNN model.
    """
    print(f"\n{'='*60}")
    print(f"TASK 3: Character Recognition")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = load_digit_classifier()
        print("✓ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    input_path = Path(image_path)
    
    # Find all building number subdirectories
    if input_path.is_dir():
        bn_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('bn')])
    else:
        print(f"ERROR: {image_path} is not a valid directory")
        return
    
    if not bn_dirs:
        print(f"No building number directories found in: {image_path}")
        return
    
    print(f"Processing {len(bn_dirs)} building number(s)...\n")
    
    total_recognized = 0
    total_characters = 0
    results = {}
    
    for bn_dir in bn_dirs:
        print(f"Processing: {bn_dir.name}")
        
        # Find all character images in this directory
        char_files = sorted(list(bn_dir.glob('c*.png')) + list(bn_dir.glob('c*.jpg')))
        
        if not char_files:
            print(f"  ✗ No character images found")
            results[bn_dir.name] = 0
            continue
        
        print(f"  Found {len(char_files)} character(s)")
        total_characters += len(char_files)
        
        # Create output directory
        output_dir = f"output/task3/{bn_dir.name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Recognize each character
        recognized_chars = []
        
        for char_file in char_files:
            # Recognize character
            predicted_digit, confidence = recognize_character(char_file, model)
            
            if predicted_digit is None:
                print(f"    ✗ Failed to recognize: {char_file.name}")
                continue
            
            # Save prediction to text file
            output_filename = char_file.stem + '.txt'
            output_path = os.path.join(output_dir, output_filename)
            
            save_output(output_path, str(predicted_digit), output_type='txt')
            
            print(f"    ✓ {char_file.name} -> {predicted_digit} (confidence: {confidence:.2%})")
            
            recognized_chars.append(predicted_digit)
            total_recognized += 1
        
        results[bn_dir.name] = len(recognized_chars)
        
        # Print recognized building number
        if recognized_chars:
            building_number = ''.join(map(str, recognized_chars))
            print(f"  → Recognized building number: {building_number}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Results per building number:")
    for bn_name, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"    {status} {bn_name}: {count} character(s) recognized")
    
    # Calculate pass status
    pass_threshold = total_characters * 0.5
    print(f"\n  Total characters: {total_characters}")
    print(f"  Total recognized: {total_recognized}")
    print(f"  Pass requirement: {pass_threshold:.0f} characters (50%)")
    print(f"  Status: {'✓ PASS' if total_recognized >= pass_threshold else '✗ FAIL'}")
    print(f"{'='*60}\n")