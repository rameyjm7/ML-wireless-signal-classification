from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import ctypes
import gc
import json
from datetime import datetime
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
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

    @abstractmethod
    def build_model(self, input_shape, num_classes):
        pass

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
        # early_stopping_custom = CustomEarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, restore_best_weights=True)

        # Add it to the list of callbacks
        # callbacks = [early_stopping_custom]
        callbacks = []


        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 20
        for epoch in range(epochs//stats_interval):
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


    def cyclical_lr(self, epoch, base_lr=1e-7, max_lr=1e-4, step_size=10):
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
        model_name = os.path.basename(self.model_path).split(".")[0]
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

    def transfer_model_with_dropout_adjustment(self, original_model, new_model_name, dropout_rates=(0.5, 0.2, 0.1)):
        """
        Transfers weights from the original model to a new model with updated dropout rates.
        """
        # Clone the structure of the original model
        new_model = clone_model(original_model)
        new_model.set_weights(original_model.get_weights())  # Set original weights
        
        # Modify dropout rates
        for layer, rate in zip(new_model.layers, dropout_rates):
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = rate
        
        # Compile the new model with the same optimizer and loss as the original
        new_model.compile(optimizer=Adam(learning_rate=1e-6),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Save new model
        new_model.save(new_model_name)
        print(f"New model saved as {new_model_name} with dropout rates: {dropout_rates}")
        return new_model