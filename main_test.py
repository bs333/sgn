import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# Load WideResNet Architecture
def wide_resnet(num_classes, depth=28, width=2):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (3, 3), padding="same")(inputs)
    for _ in range(depth // 6):
        x = layers.Conv2D(16 * width, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)  # Use Keras's ReLU layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

# Load the Test Dataset
def load_test_dataset(cifar_version, num_classes):
    if cifar_version == "cifar10":
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif cifar_version == "cifar100":
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError("cifar_version must be 'cifar10' or 'cifar100'.")

    x_test = x_test.astype(np.float32) / 255.0  # Normalize
    y_test = to_categorical(y_test, num_classes)
    return tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

def evaluate_model(checkpoint_path, cifar_version="cifar10", num_classes=10):
    # Load the model architecture
    model = wide_resnet(num_classes)

    # Load weights
    model.load_weights(checkpoint_path)
    print(f"Checkpoint loaded from: {checkpoint_path}")

    # Compile the model for evaluation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Load test dataset
    test_ds = load_test_dataset(cifar_version, num_classes)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy

if __name__ == "__main__":
    # Path to the desired checkpoint file
    checkpoint_file = "./model/final_model.weights.h5"

    # Evaluate the model on CIFAR-10
    evaluate_model(checkpoint_file, cifar_version="cifar10", num_classes=10)