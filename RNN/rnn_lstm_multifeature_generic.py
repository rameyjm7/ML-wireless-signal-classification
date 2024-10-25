from abc import ABC, abstractmethod
from datetime import datetime
import json
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


# Base Abstract Class
class BaseModulationClassifier(ABC):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        self.data_path = data_path
        self.model_path = model_path
        self.stats_path = stats_path
        self.data_pickle_path = f"{model_path}_data.pkl"  # Pickle path for preprocessed data
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

    @abstractmethod
    def prepare_data(self):
        """
        Abstract method for preparing data that must be implemented by inheriting classes.
        """
        pass

    def save_model(self):
        mpath = f'{self.model_path}'
        self.model.save(mpath, save_format='keras')
        print(f"Model saved to {mpath}")
        
    @abstractmethod
    def build_model(self, input_shape, num_classes):
        pass


    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64, use_clr=False, clr_step_size=10):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stopping]

        # Add Cyclical Learning Rate (CLR) if requested
        if use_clr:
            clr_scheduler = LearningRateScheduler(lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size))
            callbacks.append(clr_scheduler)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)

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
   
    def train_continuously(self, X_train, y_train, X_test, y_test, batch_size=64, use_clr=False, clr_step_size=10):
        try:
            epoch = 1
            while True:
                print(f"\nStarting epoch {epoch}")
                self.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=batch_size, use_clr=use_clr, clr_step_size=clr_step_size)

                epoch += 1
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.evaluate(X_test, y_test)
            self.save_stats()

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc
    

# Child class inheriting from the abstract class, implementing `prepare_data`
class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        super().__init__(data_path, model_path, stats_path)

    def set_learning_rate(self, new_lr):
        """
        Update the learning rate for the model.
        """
        self.learning_rate = new_lr
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Learning rate set to: {self.learning_rate}")

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
            
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64, use_clr=False, clr_step_size=10, augment_data = False):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        callbacks = [early_stopping]

        # Add Cyclical Learning Rate (CLR) if requested
        if use_clr:
            clr_scheduler = LearningRateScheduler(lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size))
            callbacks.append(clr_scheduler)
    
        if augment_data:
            for epoch in range(epochs):
                # Apply progressive augmentation
                X_train_augmented    = self.augment_data_progressive(X_train.copy(), epoch, epochs)
                history = self.model.fit(X_train_augmented, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)


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
 
    def compute_fft_features(self, signal):
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)
        peak_idx = np.argmax(magnitude)
        center_frequency = peak_idx
        peak_power = 20 * np.log10(magnitude[peak_idx])
        avg_power = 20 * np.log10(np.mean(magnitude))
        std_dev_power = 20 * np.log10(np.std(magnitude))
        return center_frequency, peak_power, avg_power, std_dev_power

    def compute_instantaneous_features(self, signal):
        analytic_signal = hilbert(np.real(signal))
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode='edge')
        return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

    def autocorrelation(self, signal):
        result = np.correlate(signal, signal, mode='full')
        return result[result.size // 2:]
    
    def is_digital_signal(self, autocorr_signal, threshold=0.1):
        normalized_autocorr = autocorr_signal / np.max(autocorr_signal)
        is_digital = np.any(normalized_autocorr < threshold)
        return 1 if is_digital else 0

    def compute_kurtosis(self, signal):
        """
        Compute the kurtosis of a signal.
        Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued signal.
        """
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        kurtosis = np.mean((signal - mean_signal)**4) / (std_signal**4)
        return kurtosis
    
    def compute_skewness(self, signal):
        """
        Compute the skewness of a signal.
        Skewness is a measure of the asymmetry of the probability distribution of a real-valued signal.
        """
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        skewness = np.mean((signal - mean_signal)**3) / (std_signal**3)
        return skewness
    
    def compute_spectral_energy_concentration(self, signal, center_freq_idx, bandwidth):
        """
        Compute the spectral energy concentration around the peak center frequency.
        This measures how concentrated the energy is around the peak frequency within a specified bandwidth.
        
        :param signal: The input IQ signal (real + imaginary parts)
        :param center_freq_idx: The index of the center frequency (in terms of FFT bin)
        :param bandwidth: The bandwidth (in terms of number of bins) around the center frequency
        """
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)
        
        # Select bins within the specified bandwidth around the center frequency
        lower_bound = max(0, center_freq_idx - bandwidth // 2)
        upper_bound = min(len(magnitude), center_freq_idx + bandwidth // 2)
        
        # Compute the energy concentration within the specified bandwidth
        spectral_energy = np.sum(magnitude[lower_bound:upper_bound]**2)
        total_energy = np.sum(magnitude**2)
        
        energy_concentration = spectral_energy / total_energy
        return energy_concentration

    def compute_zero_crossing_rate(self, signal):
        """
        Compute the zero-crossing rate of a signal.
        Zero-crossing rate is the rate at which the signal changes sign.
        """
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        zcr = len(zero_crossings) / len(signal)
        return zcr
    
    def compute_instantaneous_frequency_jitter(self, instantaneous_frequency):
        """
        Compute the instantaneous frequency jitter, which is the standard deviation of instantaneous frequency.
        :param instantaneous_frequency: Array of instantaneous frequency values
        """
        freq_jitter = np.std(instantaneous_frequency)
        return freq_jitter

    def cyclical_lr(self, epoch, base_lr=1e-7, max_lr=1e-5, step_size=10):
        """
        Implements cyclical learning rate.
        The learning rate cycles between base_lr and max_lr over the course of step_size epochs.
        """
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr
    
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
    
    def prepare_data(self):
        if os.path.exists(self.data_pickle_path):
            print(f"Loading prepared data from {self.data_pickle_path}")
            with open(self.data_pickle_path, 'rb') as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            return X_train, X_test, y_train, y_test

        print("Preparing data from scratch...")

        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T

                # Compute FFT features
                center_freq, peak_power, avg_power, std_dev_power = self.compute_fft_features(signal[0] + 1j * signal[1])

                # Compute instantaneous features
                instantaneous_amplitude, instantaneous_phase, instantaneous_frequency = self.compute_instantaneous_features(signal[0] + 1j * signal[1])

                # Compute autocorrelation and digital/analog flag
                autocorr_signal = self.autocorrelation(signal[0])
                is_digital = self.is_digital_signal(autocorr_signal)

                # Higher-order statistics
                kurtosis = self.compute_kurtosis(iq_signal)
                skewness = self.compute_skewness(iq_signal)

                # Spectral energy concentration
                energy_concentration = self.compute_spectral_energy_concentration(signal[0] + 1j * signal[1], center_freq, bandwidth=10)

                # Zero-crossing rate
                zcr = self.compute_zero_crossing_rate(signal[0])

                # Instantaneous frequency jitter
                freq_jitter = self.compute_instantaneous_frequency_jitter(instantaneous_frequency)

                # SNR feature
                snr_signal = np.full((128, 1), snr)

                # Append all features
                combined_signal = np.hstack([
                    iq_signal,                
                    snr_signal,               
                    np.full((128, 1), center_freq),       
                    np.full((128, 1), peak_power),        
                    np.full((128, 1), avg_power),         
                    np.full((128, 1), std_dev_power),     
                    instantaneous_amplitude.reshape(-1, 1),
                    instantaneous_phase.reshape(-1, 1),   
                    instantaneous_frequency.reshape(-1, 1),
                    np.full((128, 1), is_digital),        
                    np.full((128, 1), kurtosis),
                    np.full((128, 1), skewness),
                    np.full((128, 1), energy_concentration),
                    np.full((128, 1), zcr),
                    np.full((128, 1), freq_jitter)
                ])

                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        with open(self.data_pickle_path, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Prepared data saved to {self.data_pickle_path}")

        return X_train, X_test, y_train, y_test



# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths with the script directory as the base
data_path = os.path.join(script_dir, '..', 'RML2016.10a_dict.pkl')  # One level up from the script's directory
model_path = os.path.join(script_dir, 'rnn_lstm_multifeature_generic.keras')
stats_path = os.path.join(script_dir, f'{model_path}_stats.json')

# Usage Example
print("Data path:", data_path)
print("Model path:", model_path)
print("Stats path:", stats_path)


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
