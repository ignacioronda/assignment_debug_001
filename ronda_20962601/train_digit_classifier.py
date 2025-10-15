#!/usr/bin/env python3
"""
Train a CNN digit classifier on MNIST dataset for Task 3.

This script downloads MNIST, trains a simple CNN, and saves the model.
"""

import os

# Force CPU usage (avoid CUDA issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_digit_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple CNN for digit classification.
    
    Architecture:
    - Conv2D (32 filters) -> ReLU -> MaxPool
    - Conv2D (64 filters) -> ReLU -> MaxPool
    - Flatten
    - Dense (128) -> ReLU -> Dropout
    - Dense (num_classes) -> Softmax
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (10 for digits 0-9)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_and_preprocess_mnist():
    """
    Load MNIST dataset and preprocess for CNN training.
    
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST from Keras datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    
    # Reshape to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    Train the CNN model.
    
    Args:
        model: Keras model to train
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    print("\nCompiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print(f"\nTraining for {epochs} epochs...")
    print("(Using CPU - this may take 5-10 minutes)\n")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data
    """
    print("\nEvaluating model on test data...")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return test_accuracy


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history saved to: {save_path}")


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("Training CNN Digit Classifier for Task 3")
    print("="*60)
    
    # Create output directory for models
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Create model
    model = create_digit_cnn(input_shape=(28, 28, 1), num_classes=10)
    
    # Train model
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=10,  # Can increase to 15 for even better accuracy
        batch_size=128
    )
    
    # Evaluate model
    test_accuracy = evaluate_model(model, x_test, y_test)
    
    # Save model in multiple formats
    model_path_h5 = 'data/digit_classifier.h5'
    model_path_keras = 'data/digit_classifier.keras'
    
    model.save(model_path_h5)
    model.save(model_path_keras)
    
    print(f"\n✓ Model saved to:")
    print(f"  - {model_path_h5}")
    print(f"  - {model_path_keras}")
    
    # Plot training history
    plot_training_history(history, 'training_history.png')
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()