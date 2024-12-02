import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Log-ratio transformations
def clr_inv(p):
    z = tf.math.log(p)
    return z - tf.reduce_mean(z, axis=1)[:, tf.newaxis]

def clr_forward(z, axis=1):
    return tf.nn.softmax(z, axis=axis)

# WideResNet Architecture
def wide_resnet(num_classes, depth=28, width=2):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (3, 3), padding="same")(inputs)
    for _ in range(depth // 6):
        x = layers.Conv2D(16 * width, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    return models.Model(inputs, outputs)

# Dataset loader combining TF CIFAR datasets with custom labels
def load_combined_dataset(cifar_version, label_file, num_classes):
    """
    Combine CIFAR image data from TensorFlow with labels from a .npy file.

    Args:
        cifar_version (str): "cifar10" or "cifar100" to select the dataset.
        label_file (str): Path to the .npy file containing labels.
        num_classes (int): Number of classes (10 or 100).

    Returns:
        tf.data.Dataset: Dataset ready for training/testing.
    """
    # Load image data
    if cifar_version == "cifar10":
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    elif cifar_version == "cifar100":
        (x_train, _), _ = tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError("cifar_version must be 'cifar10' or 'cifar100'.")

    # Load labels from .npy file
    label_data = np.load(label_file, allow_pickle=True).item()
    if 'clean_label' not in label_data:
        raise ValueError("Label file must contain 'clean_label' key.")

    # Normalize image data and convert labels
    x_train = x_train.astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(label_data['clean_label'], num_classes)

    # Create TensorFlow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(64)
    return train_ds

# Gaussian noise (manual implementation)
def add_gaussian_noise(inputs, mean=0.0, stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(inputs), mean=mean, stddev=stddev)
    return inputs + noise

# Training step
@tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        smoothed_targets = clr_forward(labels)
        logit_targets = clr_inv(smoothed_targets)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(logit_targets, logits))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
def train_model(train_ds, epochs=10, num_classes=10):
    model = wide_resnet(num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (images, labels) in enumerate(train_ds):
            # Optionally, add Gaussian noise during training
            images = add_gaussian_noise(images)
            loss = train_step(model, images, labels, optimizer)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        model.save_weights(f"checkpoints/sgn_epoch_{epoch}.ckpt")
        print(f"Checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    # Paths to label files
    cifar10_label_file = "path/to/CIFAR-10_human_ordered.npy"
    cifar100_label_file = "path/to/CIFAR-100_human_ordered.npy"

    # Load CIFAR-10 dataset for training
    train_ds_cifar10 = load_combined_dataset("cifar10", cifar10_label_file, num_classes=10)

    # Train on CIFAR-10 (CIFAR-100 can be loaded similarly)
    train_model(train_ds_cifar10, epochs=10, num_classes=10)
