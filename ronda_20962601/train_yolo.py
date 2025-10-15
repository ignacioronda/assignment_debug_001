"""
Train YOLOv8 model for building number detection.
"""

from ultralytics import YOLO
from pathlib import Path

def train_building_number_detector(
    data_yaml='dataset/data.yaml',
    epochs=50,
    batch_size=4,
    img_size=640,
    model_size='n'  # n=nano (fastest), s=small, m=medium
):
    """
    Train custom YOLOv8 model for building numbers.
    
    Args:
        data_yaml: Path to data.yaml config file
        epochs: Number of training epochs
        batch_size: Batch size (use smaller if GPU memory limited)
        img_size: Input image size
        model_size: Model size (n/s/m/l/x)
    """
    
    print("="*60)
    print("Training YOLOv8 for Building Number Detection")
    print("="*60)
    
    # Load a pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        device=0,  # Use GPU 0, set to 'cpu' if no GPU
        project='runs/building_numbers',
        name='train',
        exist_ok=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {results.save_dir}")
    
    # Test on validation set
    metrics = model.val()
    print(f"\nValidation mAP50: {metrics.box.map50:.3f}")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for building numbers')
    parser.add_argument('--data', default='dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--size', default='n', choices=['n', 's', 'm'], help='Model size')
    
    args = parser.parse_args()
    
    train_building_number_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        model_size=args.size
    )