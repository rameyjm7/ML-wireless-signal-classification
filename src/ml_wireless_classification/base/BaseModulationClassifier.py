from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import ctypes
import gc
import json
from datetime import datetime
import numpy as np
import subprocess
import pickle
import threading
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.callbacks import TensorBoard

from scipy.signal import hilbert
from ml_wireless_classification.base.SignalUtils import cyclical_lr
from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping



# Base Abstract Class
class BaseModulationClassifier(ABC):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        self.name = ""
        self.data_path = data_path
        self.model_path = model_path
        self.stats_path = stats_path
        self.data_pickle_path = (
            f"{model_path}_data.pkl"  # Pickle path for preprocessed data
        )
        self.data = None
        self.label_encoder = None
        self.model = None
        self.stats = {
            "date_created": None,
            "epochs_trained": 0,
            "best_accuracy": 0,
            "current_accuracy": 0,
            "last_trained": None,
        }
        self.learning_rate = 0.0001  # Default learning rate
        self.load_stats()
        self.log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.load_data()

    def load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r") as f:
                self.stats = json.load(f)
            print(f"Loaded model stats from {self.stats_path}")
        else:
            self.stats["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_stats()

    def save_stats(self):
        with open(self.stats_path, "w") as f:
            json.dump(self.stats, f, indent=4)
        print(f"Saved model stats to {self.stats_path}")

    def load_data(self):
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f, encoding="latin1")

    @abstractmethod
    def prepare_data(self):
        """
        Abstract method for preparing data that must be implemented by inheriting classes.
        """
        pass

    def save_model(self):
        mpath = f"{self.model_path}"
        self.model.save(mpath, save_format="keras")
        print(f"Model saved to {mpath}")

        # Start TensorBoard in a background thread
    def start_tensorboard(self):
        print("Starting TensorBoard...")
        subprocess.run(["tensorboard", "--logdir", self.log_dir, "--host=0.0.0.0", "--port=6006"])

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential(
                [
                    LSTM(128, input_shape=input_shape, return_sequences=True),
                    Dropout(0.5),
                    LSTM(128, return_sequences=False),
                    Dropout(0.2),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(num_classes, activation="softmax"),
                ]
            )
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=64,
        use_clr=True,
        clr_step_size=10,
    ):
        early_stopping_custom = CustomEarlyStopping(
            monitor="val_accuracy",
            min_delta=0.01,
            patience=5,
            restore_best_weights=True,
        )

        
        tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        callbacks = [early_stopping_custom, tensorboard_callback]

        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 20
        for epoch in range(epochs // stats_interval):
            # X_train_augmented = augment_data_progressive(X_train.copy(), epoch, epochs)
            history = self.model.fit(
                np.nan_to_num(X_train, nan=0.0),
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

    def cyclical_lr(self, epoch, base_lr=1e-6, max_lr=1e-2, step_size=10):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr

    def train_continuously(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=64,
        use_clr=False,
        clr_step_size=10,
    ):
        try:
            epoch = 1
            while True:
                print(f"\nStarting epoch {epoch}")
                try:
                    self.train(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        epochs=20,
                        batch_size=batch_size,
                        use_clr=use_clr,
                        clr_step_size=clr_step_size,
                    )
                    epoch += 1

                except Exception as e:
                    print(e)
                    # Run garbage collector
                    gc.collect()
                    tf.keras.backend.clear_session()
                    pass

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.evaluate(X_test, y_test)
            self.save_stats()

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        # Generate predictions and save confusion matrix
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.save_confusion_matrix(y_test, y_pred_classes)

        return test_acc

    def get_model_name(self):
        return os.path.basename(self.model_path).split(".")[0]

    def save_confusion_matrix(self, y_true, y_pred):
        """Generates and saves a confusion matrix plot for the model's predictions."""
        conf_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=self.label_encoder.classes_
        )

        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title(f"Confusion Matrix - {os.path.basename(self.model_path)}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.get_model_name()
        save_path = os.path.join(
            common_vars.stats_dir,
            "confusion_matrix",
            f"CM_{model_name}.png",
        )
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(
            np.argmax(predictions, axis=1)
        )
        return predicted_labels

    def update_epoch_stats(self, epochs):
        """Updates the stats for epochs trained and the last trained timestamp."""
        self.stats["epochs_trained"] += epochs
        self.stats["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def update_and_save_stats(self, current_accuracy):
        """Updates stats with the current accuracy and saves the model if accuracy improves."""
        self.stats["current_accuracy"] = current_accuracy

        if current_accuracy > self.stats["best_accuracy"]:
            print(f"New best accuracy: {current_accuracy}. Saving model...")
            self.stats["best_accuracy"] = current_accuracy
            self.save_model()
        else:
            print(
                f"Current accuracy {current_accuracy} did not improve from best accuracy {self.stats['best_accuracy']}. Skipping model save."
            )

        self.save_stats()

    def setup(self):
        # Load the dataset
        self.load_data()

        # Prepare the data
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Build the model (load if it exists)
        input_shape = (
            X_train.shape[1],
            X_train.shape[2],
        )  # Time steps and features (I, Q, SNR, BW)
        num_classes = len(np.unique(y_train))  # Number of unique modulation types
        self.build_model(input_shape, num_classes)
        model_plot_path = os.path.join(
            common_vars.models_dir, "plots", f"{self.get_model_name()}.png"
        )

        # plot_model(
        #     self.model,
        #     to_file=model_plot_path,
        #     show_shapes=True,
        #     show_layer_names=True,
        # )

        return X_train, y_train, X_test, y_test

    def train_with_feature_removal(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=64,
        use_clr=True,
        clr_step_size=10,
    ):
        feature_names = ["I", "Q", "SNR", "Envelope Variance", "Inst. Freq. Variance", "Phase Jitter", "PAPR"]
        original_accuracy = 0
        feature_accuracies = {}

        # Training with all features to get baseline accuracy
        print("Training with all features...")
        history = self.train(X_train, y_train, X_test, y_test, epochs, batch_size, use_clr, clr_step_size)
        original_accuracy = max(history.history["val_accuracy"])
        print(f"Baseline accuracy with all features: {original_accuracy:.2f}")

        # Loop over each feature, remove it, and train the model
        for i in range(X_train.shape[2]):  # Loop through each feature dimension
            print(f"\nTraining without feature: {feature_names[i]}")

            # Remove the feature at index i
            X_train_reduced = np.delete(X_train, i, axis=2)
            X_test_reduced = np.delete(X_test, i, axis=2)

            # Rebuild and compile the model to ensure fresh training each time
            self.build_model(input_shape=(X_train_reduced.shape[1], X_train_reduced.shape[2]), num_classes=len(np.unique(y_train)))

            # Train the model without the selected feature
            history = self.model.fit(
                X_train_reduced,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_reduced, y_test),
                callbacks=[
                    CustomEarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, restore_best_weights=True),
                    TensorBoard(log_dir=self.log_dir, histogram_freq=1),
                    LearningRateScheduler(lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)) if use_clr else None,
                ],
            )

            # Store the accuracy for the feature removed
            reduced_accuracy = max(history.history["val_accuracy"])
            feature_accuracies[feature_names[i]] = reduced_accuracy
            print(f"Accuracy without {feature_names[i]}: {reduced_accuracy:.2f}")

        # Identify the feature whose removal had the smallest impact on accuracy
        least_useful_feature = min(feature_accuracies, key=lambda k: original_accuracy - feature_accuracies[k])
        impact = original_accuracy - feature_accuracies[least_useful_feature]
        print(f"\nLeast useful feature: {least_useful_feature} with accuracy drop of {impact:.2f}")

        return feature_accuracies, least_useful_feature

    def main(self, train=True):
        X_train, y_train, X_test, y_test = self.setup()


        # Start TensorBoard in a new thread after training is complete
        tensorboard_thread = threading.Thread(target=self.start_tensorboard)
        tensorboard_thread.daemon = True  # Daemonize thread to close with the main program
        tensorboard_thread.start()

        # allow to skip training if we only want to evaluate
        if train:
            # Train continuously with cyclical learning rates
            self.train_continuously(
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=64,
                use_clr=True,
                clr_step_size=10,
            )

        # Evaluate the model
        self.evaluate(X_test, y_test)

        #  Make predictions on the test set
        predictions = self.predict(X_test)
        print("Predicted Labels: ", predictions[:5])
        print("True Labels: ", self.label_encoder.inverse_transform(y_test[:5]))
