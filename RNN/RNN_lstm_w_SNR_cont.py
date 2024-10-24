import os
import ctypes
import json
from datetime import datetime
os.environ['LD_PRELOAD'] = '/home/dev/python/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'
ctypes.cdll.LoadLibrary("libgomp.so.1")
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

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

    def prepare_data(self):
        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T  # Combine real and imaginary parts (shape: (128, 2))

                # Append SNR as an additional feature (shape: (128, 3))
                snr_signal = np.full((128, 1), snr)  # Create an array of SNR with the same length as the signal
                combined_signal = np.hstack([iq_signal, snr_signal])  # Combine IQ and SNR into a single array (shape: (128, 3))

                X.append(combined_signal)  # Append the signal with SNR as an additional feature
                y.append(mod_type)  # Modulation type as label

        X = np.array(X)
        y = np.array(y)

        # Encode labels (modulation types)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Reshape data for LSTM: (samples, time steps, features), where features = 3 (I, Q, and SNR)
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test

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
        # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
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
            self.model.save(self.model_path)
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
model_path = 'rnn_lstm_w_SNR.h5'  # Path to save and load the model
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
classifier.set_learning_rate(1e-3)

# Train the model with Cyclical Learning Rate (CLR)
# classifier.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=64, use_clr=True, clr_step_size=10)

# Train continuously if needed
classifier.train_continuously(X_train, y_train, X_test, y_test, batch_size=64, use_clr=False, clr_step_size=10)

# Evaluate the model
classifier.evaluate(X_test, y_test)

# Optional: Make predictions on the test set
predictions = classifier.predict(X_test)
print("Predicted Labels: ", predictions[:5])
print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))
