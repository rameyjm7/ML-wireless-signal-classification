import os
import ctypes
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


class WBFM_OvR(BaseModulationClassifier):
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
                iq_signal = np.vstack([signal[0], signal[1]]).T
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([iq_signal, snr_signal])
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

    def build_model(self, input_shape = (128, 3), num_classes = 2):
        num_classes = 2
        self.model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Flatten(),  # This flattens the output from (batch_size, sequence_length, features) to (batch_size, sequence_length * features)
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ])
        return self.model

    def train_OvR(
        self,
        epochs=20,
    ):
        self.run_tensorboard = False
        X_train, y_train, X_test, y_test = self.setup()
            
        # Prepare binary labels for WBFM detection
        y_train_wbfm = (y_train == self.label_encoder.transform(['WBFM'])[0]).astype(int)
        y_test_wbfm = (y_test == self.label_encoder.transform(['WBFM'])[0]).astype(int)
        
        # Define custom class weights, with WBFM having a higher weight
        custom_weights = {0: 1, 1: 3}  # 0 = not WBFM, 1 = WBFM, increase weight of WBFM to 5
        
        # Initialize Logistic Regression with custom class weights
        wbfm_classifier = LogisticRegression(class_weight=custom_weights)
        wbfm_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train_wbfm)

        # wbfm_classifier = self.build_model()
        # wbfm_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # wbfm_classifier.fit(X_train, y_train_wbfm, epochs=10, batch_size=32)

        # Evaluate on WBFM classification task
        wbfm_accuracy = wbfm_classifier.score(X_test.reshape(X_test.shape[0], -1), y_test_wbfm)
        print(f"WBFM Detection Accuracy: {wbfm_accuracy * 100:.2f}%")

        # Predict and evaluate on WBFM classification task
        y_pred_wbfm = wbfm_classifier.predict(X_test.reshape(X_test.shape[0], -1))
        wbfm_accuracy = (y_pred_wbfm == y_test_wbfm).mean()
        print(f"WBFM Detection Accuracy: {wbfm_accuracy * 100:.2f}%")

        # Save the confusion matrix for WBFM detection
        self.save_confusion_matrix(y_test_wbfm, y_pred_wbfm)

        # Update stats
        self.update_epoch_stats(epochs)
        current_accuracy = wbfm_accuracy
        self.update_and_save_stats(current_accuracy)
        return current_accuracy
    
    def save_confusion_matrix(self, y_true, y_pred, labels=None):
        """Generates and saves a confusion matrix plot for the model's predictions."""
        conf_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        model_name = self.get_model_name()
        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            common_vars.stats_dir,
            "confusion_matrix",
            f"CM_{model_name}_{timestamp}.png",
        )
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    # set the model name
    model_name = "WBFM_OvR"
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
    classifier = WBFM_OvR(data_path, model_path, stats_path)
    # classifier.change_dropout_test()
    classifier.train_OvR()
