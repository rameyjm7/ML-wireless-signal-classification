import os
import ctypes
ctypes.cdll.LoadLibrary("libgomp.so.1")
# os.environ['LD_PRELOAD'] = '/home/dev/python/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
torch.cuda.is_available()
ret = torch.cuda.get_device_properties(0).name
print(ret)

class ModulationLSTMClassifier:
    def __init__(self, data_path, model_path="saved_model.h5"):
        """
        Initializes the classifier with the data path and optional model path.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.label_encoder = None
        self.model = None

    def load_data(self):
        """
        Loads the dataset from the pickle file.
        """
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')  # Load data with explicit 'latin1' encoding
    
    def prepare_data(self):
        """
        Prepares the data for LSTM by encoding labels and reshaping data.
        Returns the training and testing sets.
        """
        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T  # Combine real and imaginary parts
                X.append(iq_signal)
                y.append(mod_type)  # Modulation type as label

        X = np.array(X)
        y = np.array(y)

        # Encode labels (modulation types)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Reshape data for LSTM: (samples, time steps, features), where features = 2 (I and Q components)
        X_train = X_train.reshape(-1, X_train.shape[1], 2)
        X_test = X_test.reshape(-1, X_test.shape[1], 2)

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape, num_classes):
        """
        Builds and compiles the LSTM model if it doesn't exist. 
        If a saved model exists, it will load it.
        """
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
            self.model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification

            optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
            # Compile the model
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
        """
        Trains the LSTM model and saves the model after training.
        """
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        
        # Save the model after training
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test data.
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        predictions = self.model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return predicted_labels

# Usage
data_path = 'RML2016.10a_dict.pkl'
model_path = 'rnn_ADAM_LR_0p0001.h5'  # Path to save and load the model

# Initialize the classifier
classifier = ModulationLSTMClassifier(data_path, model_path)

# Load the dataset
classifier.load_data()

# Prepare the data
X_train, X_test, y_train, y_test = classifier.prepare_data()

# Build the LSTM model (load if it exists)
input_shape = (X_train.shape[1], X_train.shape[2])  # Time steps and features
num_classes = len(np.unique(y_train))  # Number of unique modulation types
classifier.build_model(input_shape, num_classes)

# Train the model
classifier.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=64)

# Evaluate the model
classifier.evaluate(X_test, y_test)

# Optional: Make predictions on the test set
predictions = classifier.predict(X_test)
print("Predicted Labels: ", predictions[:5])
print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))
