#!/usr/bin/env python3
"""
Improved YOLOv8 training script for building number detection.

This version includes:
- Better hyperparameters
- Longer training
- Better model architecture options
- Validation during training
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_building_detector_v2(
    data_yaml='dataset_v2/data.yaml',
    model_size='n',  # n=nano, s=small, m=medium
    epochs=100,
    batch_size=8,
    img_size=640,
    patience=25,
    device='cpu'
):
    """
    Train YOLOv8 model for building number detection with improved settings.
    
    Args:
        data_yaml: Path to data.yaml configuration
        model_size: Model size (n/s/m) - nano is fastest, medium is most accurate
        epochs: Number of training epochs
        batch_size: Batch size (reduce if out of memory)
        img_size: Input image size
        patience: Early stopping patience
        device: Device to use ('cpu' or '0' for GPU)
    """
    
    print("="*70)
    print("Building Number Detection - YOLOv8 Training V2")
    print("="*70)
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    print(f"\nTraining Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device}")
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    # Train the model with improved hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        save=True,
        plots=True,
        device=device,
        
        # Improved hyperparameters
        project='runs/building_numbers_v2',
        name='train',
        exist_ok=True,
        
        # Optimization
        optimizer='Adam',  # Adam often works better than SGD for small datasets
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (more aggressive)
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=15.0,  # Rotation (+/- deg)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,     # Scaling (+/- gain)
        shear=0.0,     # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,    # Vertical flip (probability)
        fliplr=0.5,    # Horizontal flip (probability)
        mosaic=1.0,    # Mosaic augmentation (probability)
        mixup=0.0,     # MixUp augmentation (probability)
        
        # Training settings
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # DFL loss gain
        
        # Validation
        val=True,
        save_period=10,  # Save checkpoint every N epochs
        
        # Other
        workers=4,
        verbose=True,
    )
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")
    
    # Save model to data directory
    output_path = 'data/building_number_detector_v2.pt'
    Path('data').mkdir(exist_ok=True)
    
    # Copy best weights
    best_model_path = Path('runs/building_numbers_v2/train/weights/best.pt')
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, output_path)
        print(f"\n✓ Best model saved to: {output_path}")
    
    print(f"\nModel location: runs/building_numbers_v2/train/weights/best.pt")
    print(f"Training plots: runs/building_numbers_v2/train/")
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 for building numbers')
    parser.add_argument('--data', default='dataset_v2/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm'], 
                       help='Model size (n=nano, s=small, m=medium)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--device', default='cpu', help='Device (cpu or 0 for GPU)')
    
    args = parser.parse_args()
    
    train_building_detector_v2(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        patience=args.patience,
        device=args.device
    )