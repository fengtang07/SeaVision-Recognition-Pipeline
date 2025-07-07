# seavision-recognition-pipeline

Deep learning system for classifying 23 species of marine animals using EfficientNetB7 and YOLOv8 object detection.

## Dataset

13,711 images across 23 classes including sharks, dolphins, whales, octopus, and other marine species.

## Model Architecture

- **Backbone**: EfficientNetB7 (ImageNet pretrained)
- **Classification Head**: Dense(128) → Dropout(0.45) → Dense(256) → Dropout(0.45) → Dense(23)
- **Object Detection**: YOLOv8n for real-time video processing
- **Optimization**: ReduceLROnPlateau with Adam optimizer


## Usage

### Training
```bash
python train_model.py --epochs 20 --batch_size 32
```

### Visualization
```bash
python visualize_results.py --model_path model.weights.h5
```

### Video Processing
```bash
python process_video.py --input video.mp4 --output processed.mp4
```

### GradCAM Analysis
```bash
python gradcam_analysis.py --model_path model.weights.h5 --image_path sample.jpg
```

## Requirements

- TensorFlow 2.x
- OpenCV
- Ultralytics YOLO
- Matplotlib
- Seaborn
- NumPy

## Installation

```bash
pip install tensorflow opencv-python ultralytics matplotlib seaborn numpy
```

## Files

- `train_model.py` - Model training script
- `visualize_results.py` - Results visualization and analysis
- `process_video.py` - Video processing with object detection
- `gradcam_analysis.py` - Model interpretability analysis

## AI-Assisted Code 

- This code is assisted by AI (Cursor)
  
## License

MIT License 
