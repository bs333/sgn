import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# Step 1: Load and Preprocess CIFAR-100 Dataset
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # Normalize images to [0, 1] range
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Convert labels to integers
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return x_train, y_train, x_test, y_test

# Step 2: Apply Label Smoothing (to avoid zeros in labels)
def apply_label_smoothing(y, num_classes, smoothing=0.1):
    y_smoothed = tf.keras.utils.to_categorical(y, num_classes)
    y_smoothed = y_smoothed * (1 - smoothing) + (smoothing / num_classes)
    return y_smoothed

# Step 3: Define the Model Architecture (e.g., ResNet18)
def build_resnet18(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Define a simple ResNet-like architecture
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Output heads for mean and variance (SGN implementation)
    mean_output = layers.Dense(num_classes)(x)       
    log_var_output = layers.Dense(num_classes)(x)   

    model = models.Model(inputs=inputs, outputs=[mean_output, log_var_output])
    return model

# Step 4: Define Custom Loss Function for SGN
def sgn_loss(y_true, mean_pred, log_var_pred):
    # Ensure y_true is the same data type as mean_pred
    y_true = tf.cast(y_true, mean_pred.dtype)

    # Calculate variance from log variance
    var_pred = tf.exp(log_var_pred)

    # Compute the negative log-likelihood of the Gaussian
    loss = 0.5 * tf.reduce_sum(((y_true - mean_pred) ** 2) / var_pred + tf.math.log(var_pred), axis=1)
    return tf.reduce_mean(loss)

# Step 5: Custom Training Loop
def train_sgn_model(model, x_train, y_train, x_val, y_val, num_classes, epochs=5, batch_size=128):
    optimizer = tf.keras.optimizers.Adam()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):
        print(f'\nStart of epoch {epoch+1}')
        # Training Loop
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                mean_pred, log_var_pred = model(x_batch_train, training=True)
                loss_value = sgn_loss(y_batch_train, mean_pred, log_var_pred)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update training metric
            train_acc_metric.update_state(y_batch_train, mean_pred)

        train_acc = train_acc_metric.result()
        print(f'Training acc over epoch: {float(train_acc):.4f}')
        train_acc_metric.reset_state()

        # Validation Loop
        for x_batch_val, y_batch_val in val_dataset:
            val_mean_pred, _ = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_mean_pred)

        val_acc = val_acc_metric.result()
        print(f'Validation acc: {float(val_acc):.4f}')
        val_acc_metric.reset_state()

# Step 6: Main Execution
def main():
    num_classes = 100
    x_train, y_train, x_test, y_test = load_cifar100()

    # Split training data into training and validation sets
    x_train, x_val = x_train[5000:], x_train[:5000]
    y_train, y_val = y_train[5000:], y_train[:5000]

    # Apply label smoothing
    y_train_smooth = apply_label_smoothing(y_train, num_classes)
    y_val_smooth = apply_label_smoothing(y_val, num_classes)

    # Build model
    model = build_resnet18(input_shape=(32, 32, 3), num_classes=num_classes)

    # Train model
    train_sgn_model(model, x_train, y_train_smooth, x_val, y_val_smooth, num_classes, epochs=50)

    # Evaluate on test set
    y_test_smooth = apply_label_smoothing(y_test, num_classes)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_smooth)).batch(128)
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for x_batch_test, y_batch_test in test_dataset:
        test_mean_pred, _ = model(x_batch_test, training=False)
        test_acc_metric.update_state(y_batch_test, test_mean_pred)

    test_acc = test_acc_metric.result()
    print(f'\nTest accuracy: {float(test_acc):.4f}')

    # Step 7: Output Results in Tabular Format
    results = {
        'Dataset': ['CIFAR-100'],
        'Test Accuracy': [float(test_acc)]
    }
    df_results = pd.DataFrame(results)
    print('\nResults:')
    print(df_results.to_string(index=False))

if __name__ == '__main__':
    main()