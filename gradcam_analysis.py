
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2

class GradCAM:
    def __init__(self, model_path, layer_name=None):
        """Initialize GradCAM with trained model"""
        
        self.model = self._load_model(model_path)
        self.layer_name = layer_name or self._find_target_layer()
        
        # Class labels
        self.class_labels = [
            'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish',
            'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers',
            'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp',
            'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale'
        ]
        
        print(f"GradCAM initialized with layer: {self.layer_name}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        
        # Data augmentation (same as training)
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
    
    def _find_target_layer(self):
        """Find the last convolutional layer"""
        
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):  # Sequential or functional layer
                for sublayer in reversed(layer.layers):
                    if len(sublayer.output_shape) == 4:  # Conv layer
                        return sublayer.name
            elif len(layer.output_shape) == 4:  # Conv layer
                return layer.name
        
        # Fallback to EfficientNet layer
        return 'top_conv'
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        
        # Load and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image_array = np.expand_dims(image, axis=0)
        
        return image, image_array
    
    def generate_gradcam(self, image_array, class_index=None):
        """Generate GradCAM heatmap"""
        
        # Get the model that maps inputs to the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]
        
        # Get gradients of the class score with respect to conv layer
        grads = tape.gradient(class_score, conv_outputs)
        
        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv layer output by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions[0].numpy()
    
    def create_superimposed_image(self, original_image, heatmap, alpha=0.6):
        """Create superimposed image with heatmap"""
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        
        # Superimpose
        superimposed = heatmap_colored * alpha + original_image * (1 - alpha)
        
        return superimposed
    
    def analyze_image(self, image_path, save_path=None, class_index=None):
        """Analyze single image with GradCAM"""
        
        # Preprocess image
        original_image, image_array = self.preprocess_image(image_path)
        
        # Generate GradCAM
        heatmap, predictions = self.generate_gradcam(image_array, class_index)
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Create superimposed image
        superimposed = self.create_superimposed_image(original_image, heatmap)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Prediction: {self.class_labels[predicted_class]}\n'
                         f'Confidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GradCAM analysis saved: {save_path}")
        
        plt.show()
        
        return heatmap, predictions
    
    def batch_analyze(self, images_dir, save_dir, num_images=15):
        """Analyze multiple images and create grid visualization"""
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        if len(image_files) < num_images:
            print(f"Warning: Only found {len(image_files)} images, using all")
            num_images = len(image_files)
        
        # Select random images
        selected_images = np.random.choice(image_files, num_images, replace=False)
        
        # Create grid
        cols = 5
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 8 * rows))
        
        for i, image_path in enumerate(selected_images):
            row = (i // cols) * 2
            col = i % cols
            
            # Preprocess and analyze
            original_image, image_array = self.preprocess_image(image_path)
            heatmap, predictions = self.generate_gradcam(image_array)
            superimposed = self.create_superimposed_image(original_image, heatmap)
            
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # Plot original
            axes[row, col].imshow(original_image)
            axes[row, col].set_title(f'{self.class_labels[predicted_class]} ({confidence:.2f})')
            axes[row, col].axis('off')
            
            # Plot GradCAM
            axes[row + 1, col].imshow(superimposed)
            axes[row + 1, col].set_title('GradCAM Overlay')
            axes[row + 1, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, rows * cols):
            row = (i // cols) * 2
            col = i % cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
        
        plt.suptitle('GradCAM Analysis - Model Attention Visualization', fontsize=16)
        plt.tight_layout()
        
        batch_save_path = os.path.join(save_dir, 'gradcam_batch_analysis.png')
        plt.savefig(batch_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Batch GradCAM analysis saved: {batch_save_path}")

def main():
    parser = argparse.ArgumentParser(description='GradCAM analysis for sea animals model')
    parser.add_argument('--model_path', required=True, help='Path to trained model weights')
    parser.add_argument('--image_path', help='Single image to analyze')
    parser.add_argument('--images_dir', help='Directory of images for batch analysis')
    parser.add_argument('--output_dir', default='gradcam_output', help='Output directory')
    parser.add_argument('--layer_name', help='Target layer name for GradCAM')
    parser.add_argument('--class_index', type=int, help='Specific class index to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize GradCAM
    print("Initializing GradCAM...")
    gradcam = GradCAM(args.model_path, args.layer_name)
    
    # Single image analysis
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image file {args.image_path} not found")
            return
        
        save_path = os.path.join(args.output_dir, 'gradcam_single_analysis.png')
        gradcam.analyze_image(args.image_path, save_path, args.class_index)
    
    # Batch analysis
    if args.images_dir:
        if not os.path.exists(args.images_dir):
            print(f"Error: Images directory {args.images_dir} not found")
            return
        
        print("Starting batch analysis...")
        gradcam.batch_analyze(args.images_dir, args.output_dir)
    
    if not args.image_path and not args.images_dir:
        print("Please provide either --image_path or --images_dir")

if __name__ == "__main__":
    import glob
    main() 
