import os
import ctypes
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping

from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.base.SignalUtils import augment_data_progressive, cyclical_lr

from tensorflow.keras.layers import Add, Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def residual_block(x, filters, kernel_size=(3, 3), stride=(1, 1)):
    # Save the input tensor for the shortcut connection
    shortcut = x
    
    # Main path: Apply two Conv2D layers with batch normalization
    x = Conv2D(filters, kernel_size, padding='same', strides=stride, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    
    # Adjust the shortcut path if the dimensions do not match
    if shortcut.shape[-1] != filters or stride != (1, 1):
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut (identity connection)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x




class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(
            data_path, model_path, stats_path
        )  # Call the base class constructor
        self.learning_rate = 0.0001  # Default learning rate
        self.name = "rnn_lstm_w_SNR"

    def prepare_data(self):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                # Separate real and imaginary parts for the IQ signal
                real_signal = signal[0]
                imag_signal = signal[1]

                # Normalize each channel separately to the range [-1, 1]
                max_real = np.max(np.abs(real_signal))
                max_imag = np.max(np.abs(imag_signal))
                real_signal = real_signal / max_real if max_real != 0 else real_signal
                imag_signal = imag_signal / max_imag if max_imag != 0 else imag_signal

                # Stack the normalized real and imaginary parts to form a (128, 2) array
                iq_array = np.vstack([real_signal, imag_signal]).T  # Shape: (256, 2)

                # Ensure iq_array matches the shape of fft_signal
                iq_array = iq_array[:128]  # Truncate if needed to match fft_signal length

                # Calculate FFT and create the combined signal
                fft_signal = np.fft.fft(signal[0] + 1j * signal[1], n=128).real  # Use real part for Conv1D
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([iq_array, fft_signal.reshape(-1, 1), snr_signal])

                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test


    def build_model(self, input_shape, num_classes):
        return self.build_model_resnet(input_shape, num_classes)
        
    def build_model_resnet(self, input_shape, num_classes):
        # Check if model already exists
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new simplified ResNet model")

            # Define the input layer
            inputs = Input(shape=(input_shape[0], input_shape[1], 1))

            # Initial Convolutional layer
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            
            # Add simplified residual blocks
            x = residual_block(x, filters=32)
            x = residual_block(x, filters=32, stride=(2, 1))  # Downsampling once

            x = residual_block(x, filters=64, stride=(2, 1))  # Further downsampling
            x = residual_block(x, filters=64)

            # Global average pooling and a single dense layer for classification
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)  # Reduced number of units
            x = Dropout(0.5)(x)

            # Output layer
            outputs = Dense(num_classes, activation='softmax')(x)

            # Define the model
            self.model = Model(inputs, outputs)

            # Compile the model
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy'])

        return self.model

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=64,
        use_clr=False,
        clr_step_size=10,
    ):
        # Define the custom early stopping callback
        early_stopping_custom = CustomEarlyStopping(
            monitor="val_accuracy",
            min_delta=0.01,
            patience=5,
            restore_best_weights=True,
        )

        # Add it to the list of callbacks
        callbacks = [early_stopping_custom]

        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 20
        for epoch in range(epochs // stats_interval):
            # X_train_augmented = augment_data_progressive(X_train.copy(), epoch, epochs)
            history = self.model.fit(
                X_train,
                y_train,
                epochs=stats_interval,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
            )

            self.update_epoch_stats(epochs)
            current_accuracy = max(history.history["val_accuracy"])
            self.update_and_save_stats(current_accuracy)

        return history
    
    def setup(self):
        # Load the dataset
        self.load_data()

        # Prepare the data
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Reshape data to add channel dimension for CNN
        # Assuming X_train and X_test are in shape (samples, height, width)
        # we need to reshape them to (samples, height, width, 1)
        X_train = X_train[..., np.newaxis]  # Adds a single channel dimension
        X_test = X_test[..., np.newaxis]    # Adds a single channel dimension

        # Build the model (load if it exists)
        input_shape = X_train.shape[1:]  # Now includes the new channel dimension
        num_classes = len(np.unique(y_train))  # Number of unique modulation types
        self.build_model(input_shape, num_classes)

        return X_train, y_train, X_test, y_test

    def cyclical_lr(self, epoch, base_lr=1e-3, max_lr=1e-2, step_size=10):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr

if __name__ == "__main__":
    # set the model name
    model_name = "ResNet"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory

    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.models_dir = os.path.join(script_dir, "models")
    model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

    # Usage Example
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Stats path:", stats_path)

    # Initialize the classifier
    classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
    # classifier.change_dropout_test()
    classifier.main()
