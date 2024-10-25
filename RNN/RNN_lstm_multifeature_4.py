import os
import ctypes
import json
from datetime import datetime
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from scipy.signal import hilbert

try:
    import torch
    torch.cuda.is_available()
    ret = torch.cuda.get_device_properties(0).name
    print(ret)
except:
    pass

class ModulationLSTMClassifier:
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        self.data_path = data_path
        self.model_path = model_path
        self.stats_path = stats_path
        self.data = None
        self.label_encoder = None
        self.model = None
        self.stats = {
            "date_created": None,
            "epochs_trained": 0,
            "best_accuracy": 0,
            "current_accuracy": 0,
            "last_trained": None
        }
        self.learning_rate = 0.0001  # Default learning rate
        self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded model stats from {self.stats_path}")
        else:
            self.stats["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_stats()

    def save_stats(self):
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Saved model stats to {self.stats_path}")

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def compute_fft_features(self, signal):
        """
        Compute the FFT to get the center frequency, peak power, average power, and std dev of power.
        """
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)

        # Find the index of the peak magnitude
        peak_idx = np.argmax(magnitude)

        # Peak frequency based on the FFT bin (normalized for now)
        center_frequency = peak_idx  # In terms of bin number. You can scale to Hz if needed.

        # Peak power in dB
        peak_power = 20 * np.log10(magnitude[peak_idx])

        # Average power in dB
        average_power = 20 * np.log10(np.mean(magnitude))

        # Standard deviation of power
        std_dev_power = 20 * np.log10(np.std(magnitude))

        return center_frequency, peak_power, average_power, std_dev_power

    def compute_instantaneous_features(self, signal):
        """
        Compute the instantaneous amplitude, phase, and frequency using the Hilbert transform.
        """
        # Compute the analytic signal using Hilbert transform
        analytic_signal = hilbert(np.real(signal))

        # Instantaneous amplitude (envelope of the signal)
        instantaneous_amplitude = np.abs(analytic_signal)

        # Instantaneous phase
        instantaneous_phase = np.angle(analytic_signal)

        # Instantaneous frequency is the derivative of the phase
        instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase))
        
        # Since the diff operation reduces the length by 1, we pad it to match the original length
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode='edge')

        return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

    def autocorrelation(self, signal):
        """
        Compute the autocorrelation of the signal.
        """
        result = np.correlate(signal, signal, mode='full')
        return result[result.size // 2:]

    def is_digital_signal(self, autocorr_signal, threshold=0.1):
        """
        Determine if the signal is digital or analog based on its autocorrelation.
        Digital signals tend to have sharper changes in their autocorrelation function.
        """
        # Find the first major drop in the autocorrelation
        normalized_autocorr = autocorr_signal / np.max(autocorr_signal)
        # If the autocorrelation drops significantly (below the threshold), we assume it's digital
        is_digital = np.any(normalized_autocorr < threshold)
        return 1 if is_digital else 0

    def compute_fft_features(self, signal):
        """
        Compute the FFT to get the center frequency, peak power, average power, and std dev of power.
        """
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)

        # Find the index of the peak magnitude
        peak_idx = np.argmax(magnitude)

        # Peak frequency based on the FFT bin (normalized for now)
        center_frequency = peak_idx  # In terms of bin number

        # Peak power in dB
        peak_power = 20 * np.log10(magnitude[peak_idx])

        # Average power and standard deviation of power
        avg_power = 20 * np.log10(np.mean(magnitude))
        std_dev_power = np.std(20 * np.log10(magnitude))

        return center_frequency, peak_power, avg_power, std_dev_power

    def compute_instantaneous_features(self, signal):
        """
        Compute instantaneous amplitude, phase, and frequency using the Hilbert transform.
        """
        analytic_signal = hilbert(signal)
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)  # Derivative of the phase

        # Pad the instantaneous frequency to match the original signal length
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode='constant')

        return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

    def prepare_data(self):
        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T  # Combine real and imaginary parts (shape: (128, 2))

                # Compute FFT features: center frequency, peak power, average power, and std dev of power
                center_freq, peak_power, avg_power, std_dev_power = self.compute_fft_features(signal[0] + 1j * signal[1])
                
                # Compute the instantaneous amplitude, phase, and frequency from the IQ signal
                instantaneous_amplitude, instantaneous_phase, instantaneous_frequency = self.compute_instantaneous_features(signal[0] + 1j * signal[1])

                # Compute autocorrelation of the real part of the signal and determine if it's digital or analog
                autocorr_signal = self.autocorrelation(signal[0])
                is_digital = self.is_digital_signal(autocorr_signal)  # 1 if digital, 0 if analog

                # Append SNR as an additional feature (shape: (128, 1))
                snr_signal = np.full((128, 1), snr)  # Create an array of SNR with the same length as the signal
                
                # Append FFT-based features and instantaneous features
                center_freq_signal = np.full((128, 1), center_freq)  # Center frequency as a feature
                peak_power_signal = np.full((128, 1), peak_power)  # Peak power in dB as a feature
                avg_power_signal = np.full((128, 1), avg_power)  # Average power in dB as a feature
                std_dev_power_signal = np.full((128, 1), std_dev_power)  # Standard deviation of power as a feature
                inst_amplitude_signal = instantaneous_amplitude.reshape(-1, 1)  # Instantaneous amplitude
                inst_phase_signal = instantaneous_phase.reshape(-1, 1)  # Instantaneous phase
                inst_frequency_signal = instantaneous_frequency.reshape(-1, 1)  # Instantaneous frequency
                is_digital_signal_feature = np.full((128, 1), is_digital)  # Whether the signal is digital or analog
                
                # Combine all features into a single array
                combined_signal = np.hstack([
                    iq_signal,                # IQ Signal (real and imaginary)
                    snr_signal,               # SNR
                    center_freq_signal,       # Center frequency of the peak
                    peak_power_signal,        # Peak power
                    avg_power_signal,         # Average power
                    std_dev_power_signal,     # Standard deviation of power
                    inst_amplitude_signal,    # Instantaneous amplitude
                    inst_phase_signal,        # Instantaneous phase
                    inst_frequency_signal,    # Instantaneous frequency
                    is_digital_signal_feature # Whether the signal is digital or analog
                ])

                X.append(combined_signal)  # Append the signal with all additional features
                y.append(mod_type)  # Modulation type as label

        X = np.array(X)
        y = np.array(y)

        # Encode labels (modulation types)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Reshape data for LSTM: (samples, time steps, features)
        # The features now include I, Q, SNR, center frequency, peak power, average power, std dev of power, instantaneous amplitude, phase, frequency, and whether it's digital or analog.
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test
    
    def augment_data_progressive(self, X, current_epoch, total_epochs, augmentation_params=None):
        """
        Gradually reduce augmentation intensity over time.
        :param X: Input data to augment
        :param current_epoch: The current epoch number
        :param total_epochs: Total number of epochs for the training
        :param augmentation_params: Dictionary of augmentation parameters (e.g., noise level, scale factor)
        :return: Augmented data
        """
        if augmentation_params is None:
            augmentation_params = {
                "noise_level": 0.001,
                "scale_range": (0.99, 1.01),
                "shift_range": (-0.01, 0.01),
                "augment_percent": 0.5  # Start augmenting 50% of the data
            }

        noise_level = augmentation_params["noise_level"]
        scale_range = augmentation_params["scale_range"]
        shift_range = augmentation_params["shift_range"]
        augment_percent = augmentation_params["augment_percent"]

        # Decrease augmentation intensity as training progresses
        scale_factor = 1 - (current_epoch / total_epochs)
        noise_level *= scale_factor
        scale_range = (1 + (scale_range[0] - 1) * scale_factor, 1 + (scale_range[1] - 1) * scale_factor)
        shift_range = (shift_range[0] * scale_factor, shift_range[1] * scale_factor)

        # Selectively augment a subset of the data
        num_samples = X.shape[0]
        num_to_augment = int(num_samples * augment_percent * scale_factor)
        indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)

        noise = np.random.normal(0, noise_level, (num_to_augment, X.shape[1], X.shape[2]))
        scale = np.random.uniform(scale_range[0], scale_range[1], (num_to_augment, X.shape[1], X.shape[2]))
        shift = np.random.uniform(shift_range[0], shift_range[1], (num_to_augment, X.shape[1], X.shape[2]))

        X[indices_to_augment] = X[indices_to_augment] * scale + noise + shift
        print(f"Data augmented progressively with noise, scaling, and shifting for {num_to_augment} samples.")
        return X

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(128, return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(num_classes, activation='softmax'))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def set_learning_rate(self, new_lr):
        """
        Update the learning rate for the model.
        """
        self.learning_rate = new_lr
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Learning rate set to: {self.learning_rate}")

    def cyclical_lr(self, epoch, base_lr=1e-5, max_lr=1e-3, step_size=10):
        """
        Implements cyclical learning rate.
        The learning rate cycles between base_lr and max_lr over the course of step_size epochs.
        """
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64, use_clr=False, clr_step_size=10):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        callbacks = [early_stopping]

        # Add Cyclical Learning Rate (CLR) if requested
        if use_clr:
            clr_scheduler = LearningRateScheduler(lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size))
            callbacks.append(clr_scheduler)

        for epoch in range(epochs):
            # Apply progressive augmentation
            X_train_augmented = self.augment_data_progressive(X_train.copy(), epoch, epochs)
            history = self.model.fit(X_train_augmented, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)

        # Update total number of epochs trained
        self.stats["epochs_trained"] += epochs
        self.stats["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_accuracy = max(history.history['val_accuracy'])
        self.stats["current_accuracy"] = current_accuracy

        # Check if current accuracy is better than best_accuracy
        if current_accuracy > self.stats["best_accuracy"]:
            print(f"New best accuracy: {current_accuracy}. Saving model...")
            self.stats["best_accuracy"] = current_accuracy
            # Save the model if the accuracy improved
            self.save_model()
        else:
            print(f"Current accuracy {current_accuracy} did not improve from best accuracy {self.stats['best_accuracy']}. Skipping model save.")

        # Save the updated stats
        self.save_stats()

        return history

    def save_model(self):
        mpath = f'{self.model_path}'
        self.model.save(mpath, save_format='keras')
        print(f"Model saved to {mpath}")

    def train_continuously(self, X_train, y_train, X_test, y_test, batch_size=64, use_clr=False, clr_step_size=10):
        try:
            epoch = 1
            while True:
                print(f"\nStarting epoch {epoch}")
                self.train(X_train, y_train, X_test, y_test, epochs=1, batch_size=batch_size, use_clr=use_clr, clr_step_size=clr_step_size)

                epoch += 1
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.evaluate(X_test, y_test)
            self.save_stats()

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return predicted_labels

    def change_optimizer(self, new_optimizer):
        self.model.compile(optimizer=new_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Optimizer updated and model recompiled.")


# Usage
data_path = '../RML2016.10a_dict.pkl'
model_path = 'rnn_lstm_multifeature_2.keras'  # Path to save and load the model
stats_path = f'{model_path}_stats.json'  # Path to save and load model stats

# Initialize the classifier
classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)

# Load the dataset
classifier.load_data()

# Prepare the data
X_train, X_test, y_train, y_test = classifier.prepare_data()

# Build the LSTM model (load if it exists)
input_shape = (X_train.shape[1], X_train.shape[2])  # Time steps and features (with SNR as additional feature)
num_classes = len(np.unique(y_train))  # Number of unique modulation types
classifier.build_model(input_shape, num_classes)

# Set the learning rate
classifier.set_learning_rate(1e-4)

# Train continuously with cyclical learning rates
classifier.train_continuously(X_train, y_train, X_test, y_test, batch_size=64, use_clr=True, clr_step_size=10)

# Evaluate the model
classifier.evaluate(X_test, y_test)

# Optional: Make predictions on the test set
predictions = classifier.predict(X_test)
print("Predicted Labels: ", predictions[:5])
print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))
