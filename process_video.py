
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time

# Sea animal classification
SEA_ANIMAL_FACTS = {
    'Clams': 'Can live over 500 years and filter water',
    'Corals': 'Are animals, not plants, and build reefs',
    'Crabs': 'Walk sideways due to leg structure',
    'Dolphin': 'Have names for each other using clicks',
    'Eel': 'Can generate up to 600 volts of electricity',
    'Fish': 'Can see colors humans cannot perceive',
    'Jelly Fish': 'Have no brain, heart, or blood',
    'Lobster': 'Were once considered poor people food',
    'Nudibranchs': 'Are sea slugs with incredible colors',
    'Octopus': 'Have three hearts and blue blood',
    'Otter': 'Hold hands while sleeping to stay together',
    'Penguin': 'Can drink saltwater and filter out salt',
    'Puffers': 'Inflate to 3x their size when threatened',
    'Sea Rays': 'Are closely related to sharks',
    'Sea Urchins': 'Walk on their spines like stilts',
    'Seahorse': 'Males carry and give birth to babies',
    'Seal': 'Can hold their breath for over 2 hours',
    'Sharks': 'Existed before trees and have no bones',
    'Shrimp': 'Can see 16 types of color vision',
    'Squid': 'Have 10 arms and can change color instantly',
    'Starfish': 'Can regenerate lost arms and have no brain',
    'Turtle_Tortoise': 'Navigate using magnetic fields',
    'Whale': 'Blue whales hearts are as big as cars'
}

class VideoProcessor:
    def __init__(self, model_path=None, yolo_model="yolov8n.pt"):
        """Initialize video processor"""
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model)
        
        # Load classification model if provided
        self.classification_model = None
        if model_path and os.path.exists(model_path):
            print("Loading classification model...")
            self.classification_model = self._load_classification_model(model_path)
        
        # Class labels
        self.class_labels = [
            'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish',
            'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers',
            'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp',
            'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale'
        ]
        
        print("Video processor initialized!")
    
    def _load_classification_model(self, model_path):
        """Load classification model"""
        
        # Data augmentation
        augment = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        
        # Base model
        base_model = tf.keras.applications.EfficientNetB7(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='max'
        )
        base_model.trainable = False
        
        # Build model
        inputs = base_model.input
        x = augment(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
        x = tf.keras.layers.Dropout(0.45)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.45)(x)
        outputs = tf.keras.layers.Dense(23, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.load_weights(model_path)
        
        return model
    
    def classify_animal(self, roi):
        """Classify the animal in the ROI"""
        if self.classification_model is None:
            return None, 0.0
        
        try:
            # Preprocess
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_array = np.expand_dims(roi_rgb, axis=0)
            
            # Predict
            predictions = self.classification_model.predict(roi_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            if confidence > 0.2:
                return self.class_labels[class_idx], confidence
            return None, 0.0
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0.0
    
    def process_video(self, input_path, output_path, confidence_threshold=0.3):
        """Process video with YOLO detection and classification"""
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        
                        if conf > confidence_threshold:
                            detection_count += 1
                            
                            # Extract ROI for classification
                            roi = frame[y1:y2, x1:x2]
                            
                            if roi.size > 0:
                                # Classify animal
                                animal_class, class_conf = self.classify_animal(roi)
                                
                                # Draw bounding box
                                color = (0, 255, 0)  # Green
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Prepare labels
                                if animal_class:
                                    label1 = f"{animal_class} ({class_conf:.2f})"
                                    fact = SEA_ANIMAL_FACTS.get(animal_class, "Marine animal")
                                    label2 = f"Fact: {fact}"
                                else:
                                    label1 = f"Marine Life ({conf:.2f})"
                                    label2 = "Detected by YOLO"
                                
                                # Calculate text size and position
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 2
                                
                                (text_width1, text_height1), _ = cv2.getTextSize(label1, font, font_scale, thickness)
                                (text_width2, text_height2), _ = cv2.getTextSize(label2, font, font_scale, thickness)
                                
                                # Position labels at bottom of bounding box
                                total_height = text_height1 + text_height2 + 10
                                max_width = max(text_width1, text_width2) + 10
                                
                                # Ensure labels stay within frame
                                label_y = min(y2, height - total_height)
                                label_x = min(x1, width - max_width)
                                
                                # Draw background rectangle
                                cv2.rectangle(frame,
                                            (label_x, label_y),
                                            (label_x + max_width, label_y + total_height),
                                            color, -1)
                                
                                # Draw text
                                cv2.putText(frame, label1,
                                          (label_x + 5, label_y + text_height1 + 2),
                                          font, font_scale, (255, 255, 255), thickness)
                                cv2.putText(frame, label2,
                                          (label_x + 5, label_y + text_height1 + text_height2 + 7),
                                          font, font_scale, (255, 255, 255), thickness)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {fps_current:.1f} FPS")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\nProcessing complete!")
        print(f"Frames processed: {frame_count}")
        print(f"Detections: {detection_count}")
        print(f"Processing time: {total_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Output saved: {output_path} ({file_size:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description='Process video with sea animal detection')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--model', help='Classification model weights path')
    parser.add_argument('--yolo', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor(args.model, args.yolo)
    
    # Process video
    processor.process_video(args.input, args.output, args.confidence)

if __name__ == "__main__":
    main() 
