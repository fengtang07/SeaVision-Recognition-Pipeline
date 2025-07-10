#!/usr/bin/env python3
"""
Sea Animals Classification Pipeline
Deep learning pipeline for marine animal species classification using EfficientNetB7


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import required TensorFlow modules
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# Version and metadata
__version__ = "1.0.0"
__author__ = "Feng Tang"

class SeaAnimalsClassifier:
    """
     Sea Animals Classification Pipeline
    
    A comprehensive deep learning pipeline for classifying marine animal species
    using state-of-the-art computer vision techniques.
    
    Key Features:
    - EfficientNetB7 backbone with ImageNet pretraining
    - Two-stage transfer learning strategy
    - Focal loss for handling class imbalance
    - Advanced data augmentation
    - Mixed precision training support
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, config=None):
        """
        Initialize the Sea Animals Classifier
        
        Args:
            config (dict, optional): Configuration dictionary. If None, uses default config.
        """
        self.config = config or self._get_default_config()
        self._setup_environment()
        self._initialize_variables()
        
        print(" Sea Animals Classification Pipeline v{} Initialized".format(__version__))
        print(f" Target Classes: {self.config['NUM_CLASSES']}")
        print(f" Batch Size: {self.config['BATCH_SIZE']}")
        print(f"Mixed Precision: {self.config['MIXED_PRECISION']}")

    def _get_default_config(self):
        """Get default configuration settings"""
        return {
            # Model Configuration
            'NUM_CLASSES': 23,
            'TARGET_SIZE': (224, 224),
            'BASE_MODEL': 'EfficientNetB7',
            'DROPOUT_RATE': 0.45,
            'L2_REGULARIZATION': 0.001,
            
            # Training Configuration
            'BATCH_SIZE': 32,
            'INITIAL_EPOCHS': 15,
            'FINE_TUNE_EPOCHS': 10,
            'INITIAL_LR': 0.001,
            'FINE_TUNE_LR': 0.0001,
            
            # Data Pipeline Configuration
            'BUFFER_SIZE': 2000,
            'CACHE_DATASET': True,
            'PREFETCH_DATA': True,
            'MIXED_PRECISION': True,
            
            # Data Augmentation Configuration
            'AUGMENTATION': {
                'horizontal_flip': True,
                'vertical_flip': True,
                'rotation': True,
                'zoom_range': 0.15,
                'contrast_range': 0.2,
                'brightness_range': 0.1,
                'saturation_range': 0.1,
                'hue_range': 0.05
            },
            
            # Callback Configuration
            'EARLY_STOPPING_PATIENCE': 5,
            'REDUCE_LR_PATIENCE': 3,
            'REDUCE_LR_FACTOR': 0.2,
            'MIN_LR': 1e-7,
            
            # Focal Loss Configuration
            'FOCAL_ALPHA': 0.25,
            'FOCAL_GAMMA': 2.0,
            
            # Output Configuration
            'MODEL_NAME': 'sea_animals_classifier',
            'SAVE_BEST_ONLY': True,
            'VERBOSE_TRAINING': 1
        }

    def _setup_environment(self):
        """Setup TensorFlow environment and optimizations"""
        # Configure GPU settings (if available)
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🔧 GPU Memory Growth Enabled for {len(gpus)} GPU(s)")
            else:
                print("🔧 Running on CPU")
        except Exception as e:
            print(f"⚠ GPU setup warning: {e}")
        
        # Enable mixed precision if requested
        if self.config['MIXED_PRECISION']:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("⚡ Mixed precision enabled for faster training")
            except Exception as e:
                print(f"⚠ Mixed precision not available: {e}")
                self.config['MIXED_PRECISION'] = False
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def _initialize_variables(self):
        """Initialize instance variables"""
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_labels = None
        self.history_stage1 = None
        self.history_stage2 = None
        self.training_stats = {}

    def load_dataset(self, dataset_path, class_labels=None):
        """
        Load and prepare the marine animals dataset
        
        Args:
            dataset_path (str): Path to dataset directory
            class_labels (list, optional): List of class names. If None, auto-detected.
            
        Returns:
            tuple: (train_df, val_df, test_df) DataFrames
        """
        print(f"\n Loading dataset from: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Collect image files and labels
        filepaths = []
        labels = []
        
        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        
        for ext in image_extensions:
            for img_path in dataset_path.rglob(ext):
                try:
                    # Extract class label from parent directory
                    class_name = img_path.parent.name
                    filepaths.append(str(img_path))
                    labels.append(class_name)
                except Exception as e:
                    print(f"⚠ Skipping {img_path}: {e}")
                    continue
        
        if not filepaths:
            raise ValueError(f"No images found in {dataset_path}")
        
        # Create DataFrame
        df = pd.DataFrame({'filepath': filepaths, 'label': labels})
        
        # Filter to specified classes or detect automatically
        if class_labels is None:
            self.class_labels = sorted(df['label'].unique())
            print(f" Auto-detected {len(self.class_labels)} classes")
        else:
            self.class_labels = class_labels
            df = df[df['label'].isin(class_labels)]
            print(f" Using {len(self.class_labels)} specified classes")
        
        self.config['NUM_CLASSES'] = len(self.class_labels)
        
        # Create stratified splits
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df['label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
        )
        
        print(f"Dataset loaded successfully:")
        print(f"   Total images: {len(df):,}")
        print(f"   Training: {len(train_df):,}")
        print(f"   Validation: {len(val_df):,}")
        print(f"   Testing: {len(test_df):,}")
        print(f"   Classes: {', '.join(self.class_labels[:5])}{'...' if len(self.class_labels) > 5 else ''}")
        
        return train_df, val_df, test_df

    def create_data_pipeline(self, train_df, val_df, test_df):
        """
        Create optimized tf.data pipeline for training
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data  
            test_df (pd.DataFrame): Test data
        """
        print("\n Creating optimized data pipeline...")
        
        # Create label mapping
        label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        
        # Create datasets
        self.train_ds = self._create_dataset(train_df, label_to_index, training=True)
        self.val_ds = self._create_dataset(val_df, label_to_index, training=False)
        self.test_ds = self._create_dataset(test_df, label_to_index, training=False)
        
        print(" Data pipeline created successfully")

    def _create_dataset(self, df, label_to_index, training=False):
        """Create tf.data.Dataset with preprocessing pipeline"""
        # Convert labels to one-hot
        labels = [label_to_index[label] for label in df['label']]
        one_hot_labels = tf.one_hot(labels, self.config['NUM_CLASSES'])
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((df['filepath'].values, one_hot_labels))
        
        # Load and preprocess images
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation for training
        if training:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(self.config['BUFFER_SIZE'])
        
        # Apply EfficientNet preprocessing
        dataset = dataset.map(
            self._apply_efficientnet_preprocessing,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Performance optimizations
        if self.config['CACHE_DATASET']:
            dataset = dataset.cache()
        
        dataset = dataset.batch(self.config['BATCH_SIZE'])
        
        if self.config['PREFETCH_DATA']:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def _load_and_preprocess_image(self, filepath, label):
        """Load and basic preprocessing of images"""
        image = tf.io.read_file(filepath)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.config['TARGET_SIZE'])
        return image, label

    def _augment_image(self, image, label):
        """Apply data augmentation transformations"""
        aug_config = self.config['AUGMENTATION']
        
        # Random flips
        if aug_config['horizontal_flip']:
            image = tf.image.random_flip_left_right(image)
        if aug_config['vertical_flip']:
            image = tf.image.random_flip_up_down(image)
        
        # Random rotation (90-degree increments)
        if aug_config['rotation']:
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k)
        
        # Random zoom/crop
        if aug_config['zoom_range'] > 0:
            crop_factor = tf.random.uniform([], 1.0 - aug_config['zoom_range'], 1.0)
            crop_size = tf.cast(self.config['TARGET_SIZE'][0] * crop_factor, tf.int32)
            image = tf.image.random_crop(image, [crop_size, crop_size, 3])
            image = tf.image.resize(image, self.config['TARGET_SIZE'])
        
        # Color augmentations
        if aug_config['contrast_range'] > 0:
            image = tf.image.random_contrast(
                image, 
                1.0 - aug_config['contrast_range'], 
                1.0 + aug_config['contrast_range']
            )
        
        if aug_config['brightness_range'] > 0:
            image = tf.image.random_brightness(image, aug_config['brightness_range'])
        
        if aug_config['saturation_range'] > 0:
            image = tf.image.random_saturation(
                image,
                1.0 - aug_config['saturation_range'],
                1.0 + aug_config['saturation_range']
            )
        
        if aug_config['hue_range'] > 0:
            image = tf.image.random_hue(image, aug_config['hue_range'])
        
        # Ensure valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    def _apply_efficientnet_preprocessing(self, image, label):
        """Apply EfficientNet-specific preprocessing"""
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    def build_model(self):
        """Build the EfficientNetB7-based classification model"""
        print(f"\n🏗️ Building {self.config['BASE_MODEL']} model...")
        
        # Load pre-trained base model
        base_model = getattr(tf.keras.applications, self.config['BASE_MODEL'].lower())(
            input_shape=self.config['TARGET_SIZE'] + (3,),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze during initial training
        
        # Build model architecture
        inputs = layers.Input(shape=self.config['TARGET_SIZE'] + (3,))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config['DROPOUT_RATE'])(x)
        
        # Additional dense layer with regularization
        x = Dense(
            512, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.config['L2_REGULARIZATION'])
        )(x)
        x = Dropout(self.config['DROPOUT_RATE'])(x)
        
        # Output layer (ensure float32 for mixed precision)
        if self.config['MIXED_PRECISION']:
            outputs = Dense(self.config['NUM_CLASSES'], activation='softmax', dtype='float32')(x)
        else:
            outputs = Dense(self.config['NUM_CLASSES'], activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name=self.config['MODEL_NAME'])
        
        print(f" Model built successfully:")
        print(f"    Parameters: {self.model.count_params():,}")
        print(f"    Architecture: {self.config['BASE_MODEL']} + Custom Head")

    def focal_loss(self, alpha=None, gamma=None):
        """
        Focal Loss implementation for handling class imbalance
        
        Args:
            alpha (float): Weighting factor for rare class (default from config)
            gamma (float): Focusing parameter (default from config)
        """
        if alpha is None:
            alpha = self.config['FOCAL_ALPHA']
        if gamma is None:
            gamma = self.config['FOCAL_GAMMA']
        
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate cross entropy
            ce = -y_true * tf.math.log(y_pred)
            
            # Calculate focal weight
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
            
            # Apply focal weight
            focal_loss = focal_weight * ce
            return tf.reduce_sum(focal_loss, axis=1)
        
        return focal_loss_fn

    def train(self):
        """
        Execute two-stage training process
        
        Returns:
            dict: Training statistics and metrics
        """
        print("\n Starting two-stage training process...")
        
        # Setup callbacks
        callbacks = self._get_callbacks()
        
        # Stage 1: Train classifier head
        print(f"\n STAGE 1: Training classifier head ({self.config['INITIAL_EPOCHS']} epochs)")
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['INITIAL_LR']),
            loss=self.focal_loss(),
            metrics=['accuracy']
        )
        
        start_time = time.time()
        self.history_stage1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config['INITIAL_EPOCHS'],
            callbacks=callbacks,
            verbose=self.config['VERBOSE_TRAINING']
        )
        stage1_time = time.time() - start_time
        
        # Stage 2: Fine-tune entire model
        print(f"\n🔧 STAGE 2: Fine-tuning entire model ({self.config['FINE_TUNE_EPOCHS']} epochs)")
        
        # Unfreeze base model
        self.model.get_layer(self.config['BASE_MODEL'].lower()).trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['FINE_TUNE_LR']),
            loss=self.focal_loss(),
            metrics=['accuracy']
        )
        
        start_time = time.time()
        self.history_stage2 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config['FINE_TUNE_EPOCHS'],
            callbacks=callbacks,
            verbose=self.config['VERBOSE_TRAINING']
        )
        stage2_time = time.time() - start_time
        
        # Calculate training statistics
        total_time = stage1_time + stage2_time
        self.training_stats = {
            'stage1_epochs': self.config['INITIAL_EPOCHS'],
            'stage2_epochs': self.config['FINE_TUNE_EPOCHS'],
            'total_time_minutes': total_time / 60,
            'stage1_best_val_acc': max(self.history_stage1.history['val_accuracy']),
            'stage2_best_val_acc': max(self.history_stage2.history['val_accuracy']),
            'model_name': self.config['MODEL_NAME'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n Training completed successfully!")
        print(f"    Total time: {total_time/60:.1f} minutes")
        print(f"    Best validation accuracy: {self.training_stats['stage2_best_val_acc']:.4f}")
        
        return self.training_stats

    def _get_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['EARLY_STOPPING_PATIENCE'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['REDUCE_LR_FACTOR'],
                patience=self.config['REDUCE_LR_PATIENCE'],
                min_lr=self.config['MIN_LR'],
                verbose=1
            )
        ]
        
        if self.config['SAVE_BEST_ONLY']:
            callbacks.append(
                ModelCheckpoint(
                    f"{self.config['MODEL_NAME']}_best.weights.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            )
        
        return callbacks

    def evaluate(self):
        """
        Evaluate model on test dataset
        
        Returns:
            dict: Evaluation metrics
        """
        print("\n Evaluating model on test dataset...")
        
        # Basic evaluation
        test_loss, test_accuracy = self.model.evaluate(self.test_ds, verbose=1)
        
        # Detailed predictions for classification report
        predictions = self.model.predict(self.test_ds, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Extract true labels from test dataset
        true_labels = []
        for _, labels in self.test_ds.unbatch():
            true_labels.append(np.argmax(labels.numpy()))
        true_labels = np.array(true_labels)
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=self.class_labels,
            output_dict=True
        )
        
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': report,
            'num_test_samples': len(true_labels)
        }
        
        print(f" Evaluation completed:")
        print(f"    Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        return evaluation_results

    def save_model(self, filepath=None):
        """
        Save the trained model
        
        Args:
            filepath (str, optional): Path to save model. If None, uses default naming.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.config['MODEL_NAME']}_{timestamp}.weights.h5"
        
        self.model.save_weights(filepath)
        print(f" Model saved to: {filepath}")
        
        # Save configuration
        config_path = filepath.replace('.weights.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"⚙ Configuration saved to: {config_path}")

    def visualize_training_history(self, save_path=None):
        """
        Visualize training history
        
        Args:
            save_path (str, optional): Path to save visualization
        """
        if self.history_stage1 is None:
            print("⚠️ No training history available")
            return
        
        # Combine training histories
        combined_acc = (self.history_stage1.history['accuracy'] + 
                       (self.history_stage2.history['accuracy'] if self.history_stage2 else []))
        combined_val_acc = (self.history_stage1.history['val_accuracy'] + 
                           (self.history_stage2.history['val_accuracy'] if self.history_stage2 else []))
        combined_loss = (self.history_stage1.history['loss'] + 
                        (self.history_stage2.history['loss'] if self.history_stage2 else []))
        combined_val_loss = (self.history_stage1.history['val_loss'] + 
                            (self.history_stage2.history['val_loss'] if self.history_stage2 else []))
        
        epochs = range(1, len(combined_acc) + 1)
        stage1_end = len(self.history_stage1.history['accuracy'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy
        ax1.plot(epochs, combined_acc, 'b-', label='Training accuracy', linewidth=2)
        ax1.plot(epochs, combined_val_acc, 'r-', label='Validation accuracy', linewidth=2)
        if self.history_stage2:
            ax1.axvline(x=stage1_end, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training loss
        ax2.plot(epochs, combined_loss, 'b-', label='Training loss', linewidth=2)
        ax2.plot(epochs, combined_val_loss, 'r-', label='Validation loss', linewidth=2)
        if self.history_stage2:
            ax2.axvline(x=stage1_end, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training summary
        ax3.axis('off')
        summary_text = f"""
        Training Summary:
        
        Model: {self.config['BASE_MODEL']}
        Total Epochs: {len(combined_acc)}
        Stage 1 Epochs: {self.config['INITIAL_EPOCHS']}
        Stage 2 Epochs: {self.config['FINE_TUNE_EPOCHS'] if self.history_stage2 else 0}
        
        Final Validation Accuracy: {combined_val_acc[-1]:.4f}
        Best Validation Accuracy: {max(combined_val_acc):.4f}
        
        Training Time: {self.training_stats.get('total_time_minutes', 0):.1f} minutes
        """
        ax3.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace', 
                transform=ax3.transAxes, verticalalignment='center')
        
        # Configuration details
        ax4.axis('off')
        config_text = f"""
        Configuration:
        
        Batch Size: {self.config['BATCH_SIZE']}
        Initial LR: {self.config['INITIAL_LR']}
        Fine-tune LR: {self.config['FINE_TUNE_LR']}
        Dropout Rate: {self.config['DROPOUT_RATE']}
        L2 Regularization: {self.config['L2_REGULARIZATION']}
        
        Mixed Precision: {self.config['MIXED_PRECISION']}
        Focal Loss: α={self.config['FOCAL_ALPHA']}, γ={self.config['FOCAL_GAMMA']}
        """
        ax4.text(0.1, 0.5, config_text, fontsize=11, fontfamily='monospace',
                transform=ax4.transAxes, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main training pipeline execution"""
    parser = argparse.ArgumentParser(description='Sea Animals Classification Pipeline')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs_stage1', type=int, default=15,
                       help='Number of epochs for stage 1 training')
    parser.add_argument('--epochs_stage2', type=int, default=10,
                       help='Number of epochs for stage 2 fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--model_name', type=str, default='sea_animals_classifier',
                       help='Name for saved model')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training')
    
    args = parser.parse_args()
    
    # Create custom configuration
    config = {
        'NUM_CLASSES': 23,  # Will be updated based on dataset
        'TARGET_SIZE': (224, 224),
        'BASE_MODEL': 'EfficientNetB7',
        'DROPOUT_RATE': 0.4,
        'L2_REGULARIZATION': 0.001,
        'BATCH_SIZE': args.batch_size,
        'INITIAL_EPOCHS': args.epochs_stage1,
        'FINE_TUNE_EPOCHS': args.epochs_stage2,
        'INITIAL_LR': args.learning_rate,
        'FINE_TUNE_LR': args.learning_rate * 0.1,
        'MIXED_PRECISION': args.mixed_precision,
        'MODEL_NAME': args.model_name,
        'BUFFER_SIZE': 2000,
        'CACHE_DATASET': True,
        'PREFETCH_DATA': True,
        'AUGMENTATION': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation': True,
            'zoom_range': 0.15,
            'contrast_range': 0.2,
            'brightness_range': 0.1,
            'saturation_range': 0.1,
            'hue_range': 0.05
        },
        'EARLY_STOPPING_PATIENCE': 5,
        'REDUCE_LR_PATIENCE': 3,
        'REDUCE_LR_FACTOR': 0.2,
        'MIN_LR': 1e-7,
        'FOCAL_ALPHA': 0.25,
        'FOCAL_GAMMA': 2.0,
        'SAVE_BEST_ONLY': True,
        'VERBOSE_TRAINING': 1
    }
    
    try:
        # Initialize classifier
        classifier = SeaAnimalsClassifier(config)
        
        # Load and prepare dataset
        train_df, val_df, test_df = classifier.load_dataset(args.dataset_path)
        classifier.create_data_pipeline(train_df, val_df, test_df)
        
        # Build and train model
        classifier.build_model()
        training_stats = classifier.train()
        
        # Evaluate model
        evaluation_results = classifier.evaluate()
        
        # Save model and results
        classifier.save_model()
        classifier.visualize_training_history(f"{args.model_name}_training_history.png")
        
        # Save comprehensive results
        results = {
            'training_stats': training_stats,
            'evaluation_results': evaluation_results,
            'config': config,
            'class_labels': classifier.class_labels
        }
        
        results_path = f"{args.model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n Training pipeline completed successfully!")
        print(f" Final test accuracy: {evaluation_results['test_accuracy']:.4f}")
        print(f" Results saved to: {results_path}")
        
    except Exception as e:
        print(f" Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 
