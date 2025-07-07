

import os
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_model(num_classes=23):
    """Create EfficientNetB7 model with custom head"""
    
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
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_dataset(data_dir, batch_size=32, img_size=224):
    """Load and prepare dataset"""
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def plot_training_history(history, save_path):
    """Plot training curves"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Learning Rate
    if 'lr' in history.history:
        ax3.plot(history.history['lr'])
        ax3.set_title('Learning Rate')
        ax3.set_ylabel('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_yscale('log')
        ax3.grid(True)
    
    # Top-5 Accuracy (if available)
    if 'top_5_accuracy' in history.history:
        ax4.plot(history.history['top_5_accuracy'], label='Training')
        ax4.plot(history.history['val_top_5_accuracy'], label='Validation')
        ax4.set_title('Top-5 Accuracy')
        ax4.set_ylabel('Top-5 Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train sea animals classification model')
    parser.add_argument('--data_dir', default='data', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_ds, val_ds = load_dataset(args.data_dir, args.batch_size)
    
    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f'sea_animals_model_{timestamp}.weights.h5')
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f'sea_animals_model_final_{timestamp}.weights.h5')
    model.save_weights(final_model_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, f'training_history_{timestamp}.png')
    plot_training_history(history, plot_path)
    
    # Evaluate on validation set
    print("Evaluating model...")
    val_loss, val_accuracy, val_top5 = model.evaluate(val_ds, verbose=0)
    
    print(f"\nTraining completed!")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Top-5 Accuracy: {val_top5:.4f}")
    print(f"Best model saved: {model_path}")
    print(f"Final model saved: {final_model_path}")
    print(f"Training plots saved: {plot_path}")

if __name__ == "__main__":
    main() 
