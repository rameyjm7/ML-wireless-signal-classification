from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import ctypes
import gc
from datetime import datetime
import numpy as np
import subprocess
import pickle
import threading
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.callbacks import TensorBoard
import seaborn as sns

from scipy.signal import hilbert
from ml_wireless_classification.base.SignalUtils import cyclical_lr
from ml_wireless_classification.base.CommonVars import common_vars, RUN_MODE
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
        self.run_tensorboard = True

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
                class_weight=self.class_weights_dict
            )

            self.update_epoch_stats(epochs)
            current_accuracy = max(history.history["val_accuracy"])
            self.update_and_save_stats(current_accuracy)

        return history

    def cyclical_lr(self, epoch, base_lr=1e-4, max_lr=1e-2, step_size=10):
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
        # Evaluate overall test accuracy
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Save confusion matrix
        self.save_confusion_matrix(y_test, y_pred_classes)
        
        # Calculate and print accuracy per class
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        print("\nPer-Class Accuracy:")
        for idx, accuracy in enumerate(class_accuracies):
            print(f"Class {self.label_encoder.inverse_transform([idx])[0]}: {accuracy * 100:.2f}%")
        
        # Optional: Print detailed classification report (includes precision, recall, f1-score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.label_encoder.classes_))

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

    def update_and_save_stats(self, new_stats: dict):
        """
        Updates stats with the provided new data and saves the model if accuracy improves.

        Parameters:
        - new_stats (dict): A dictionary containing the new statistics to be updated.

        Expected keys in new_stats:
        - "current_accuracy" (float): The latest accuracy.
        - "best_accuracy" (float, optional): The best accuracy recorded so far.
        """

        # Update the stats dictionary with new data
        for key, value in new_stats.items():
            self.stats[key] = value

        current_accuracy = self.stats.get("current_accuracy", 0)
        best_accuracy = self.stats.get("best_accuracy", 0)

        # Check if the current accuracy is better than the best accuracy
        if current_accuracy > best_accuracy:
            print(f"New best accuracy: {current_accuracy:.2f}. Saving model...")
            self.stats["best_accuracy"] = current_accuracy
            self.save_model()
        else:
            print(
                f"Current accuracy {current_accuracy:.2f} did not improve from best accuracy {best_accuracy:.2f}. Skipping model save."
            )

        # Save the updated stats
        self.save_stats()
    

    def apply_class_weighting(self, y_train):
        # Calculate class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        self.class_weights_dict = dict(enumerate(class_weights))
        # Increase the weight for WBFM by a factor (e.g., 2)
        focus_factor = 2  # Increase this to focus more on WBFM
        self.class_weights_dict[10] *= focus_factor

    def setup(self):
        # Load the dataset
        self.load_data()

        # Prepare the data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        self.apply_class_weighting(y_train)

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

        if self.run_tensorboard:
            self.setup_tensorboard()
        return X_train, y_train, X_test, y_test


    def setup_tensorboard(self):
        tensorboard_thread = threading.Thread(target=self.start_tensorboard)
        tensorboard_thread.daemon = True  # Daemonize thread to close with the main program
        tensorboard_thread.start()

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

    def main(self, mode : RUN_MODE):
        X_train, y_train, X_test, y_test = self.setup()

        # allow to skip training if we only want to evaluate
        if mode == RUN_MODE.TRAIN_CONTINUOUSLY:
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
        elif mode == RUN_MODE.TRAIN:
            # Train continuously with cyclical learning rates
            self.train(
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=64,
                use_clr=True,
                clr_step_size=10,
            )
        elif mode == RUN_MODE.EVALUATE_ONLY:
            print("Evaluating only...")
                 
        # Evaluate the model
        self.evaluate(X_test, y_test)

        #  Make predictions on the test set
        predictions = self.predict(X_test)
        print("Predicted Labels: ", predictions[:5])
        print("True Labels: ", self.label_encoder.inverse_transform(y_test[:5]))
        
        stats_dict = self.plot_all_analysis(X_test, y_test, self.label_encoder, False)
        self.update_and_save_stats(stats_dict)


    def augment_wbfm_samples(self, X, y, target_class, augmentation_factor=2):
        # Duplicate WBFM samples and add minor variations
        wbfm_indices = np.where(y == target_class)[0]
        augmented_X, augmented_y = [], []

        for idx in wbfm_indices:
            for _ in range(augmentation_factor):
                noise = np.random.normal(0, 0.005, X[idx].shape)
                augmented_X.append(X[idx] + noise)
                augmented_y.append(y[idx])

        return np.concatenate((X, np.array(augmented_X))), np.concatenate((y, np.array(augmented_y)))

    def wbfm_fine_tuning(self, augment = True):
        # Set up the training and testing data
        X_train, y_train, X_test, y_test = self.setup()

        # Filter dataset for WBFM and a few similar classes
        target_classes = ['WBFM', 'AM-DSB', 'QPSK']
        filtered_indices = np.isin(y_train, self.label_encoder.transform(target_classes))
        X_train_filtered = X_train[filtered_indices]
        y_train_filtered = y_train[filtered_indices]

        if augment:
            # Apply augmentation for WBFM samples in the filtered dataset
            wbfm_class_label = self.label_encoder.transform(['WBFM'])[0]
            X_train_filtered, y_train_filtered = self.augment_wbfm_samples(X_train_filtered, y_train_filtered, target_class=wbfm_class_label)

        # Fine-tune the model on the filtered and augmented dataset
        self.train_continuously(
            X_train_filtered,
            y_train_filtered,
            X_test,
            y_test,
            batch_size=64,
            use_clr=True       # Cyclical learning rate
        )

        # Evaluate the model performance
        self.evaluate(X_test, y_test)

    def plot_all_analysis(self, X_test, y_test, label_encoder, show_plot=False):
        """
        Combines all analysis and visualization into a single function for a single-branch model.
        Organizes the plots in a single figure with a dynamic grid layout.
        Saves the figure to the specified directory with the model name as the file name.
        Returns a dictionary of statistics such as accuracy over 5 dB and accuracy per SNR.
        """
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        common_vars.stats_dir = os.path.join(script_dir, "..", "stats")

        model = self.model

        # --- Confusion Matrix for All SNR Levels ---
        y_pred = np.argmax(model.predict(X_test, verbose=False), axis=1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        overall_accuracy = accuracy_score(y_test, y_pred) * 100

        # --- Confusion Matrix for SNR > 5 dB ---
        snr_above_5_indices = np.where(X_test[:, :, 2].mean(axis=1) > 5)
        X_test_snr_above_5 = X_test[snr_above_5_indices]
        y_test_snr_above_5 = y_test[snr_above_5_indices]

        if len(X_test_snr_above_5) > 0:
            y_pred_snr_above_5 = np.argmax(model.predict(X_test_snr_above_5, verbose=False), axis=1)
            conf_matrix_snr_above_5 = confusion_matrix(y_test_snr_above_5, y_pred_snr_above_5)
            accuracy_over_5dB = accuracy_score(y_test_snr_above_5, y_pred_snr_above_5) * 100
        else:
            conf_matrix_snr_above_5 = None
            accuracy_over_5dB = None

        # --- Accuracy vs. SNR ---
        unique_snrs = sorted(set(X_test[:, :, 2].mean(axis=1)))
        accuracy_per_snr = []
        for snr in unique_snrs:
            snr_indices = np.where(X_test[:, :, 2].mean(axis=1) == snr)
            X_snr = X_test[snr_indices]
            y_snr = y_test[snr_indices]
            if len(y_snr) > 0:
                y_pred_snr = np.argmax(model.predict(X_snr, verbose=0), axis=1)
                accuracy_per_snr.append(accuracy_score(y_snr, y_pred_snr) * 100)
            else:
                accuracy_per_snr.append(np.nan)

        peak_accuracy = max([acc for acc in accuracy_per_snr if not np.isnan(acc)])
        peak_snr = unique_snrs[accuracy_per_snr.index(peak_accuracy)]

        # --- Accuracy vs. SNR per Modulation Type ---
        unique_modulations = label_encoder.classes_
        modulation_traces = []
        for mod_index, mod in enumerate(unique_modulations):
            accuracies = []
            for snr in unique_snrs:
                mod_snr_indices = np.where(
                    (y_test == mod_index) & (X_test[:, :, 2].mean(axis=1) == snr)
                )
                X_mod_snr = X_test[mod_snr_indices]
                y_mod_snr = y_test[mod_snr_indices]
                if len(y_mod_snr) > 0:
                    y_pred_mod_snr = np.argmax(model.predict(X_mod_snr, verbose=False), axis=1)
                    accuracies.append(accuracy_score(y_mod_snr, y_pred_mod_snr) * 100)
                else:
                    accuracies.append(np.nan)
            modulation_traces.append((mod, accuracies))

        # --- Create Subplots ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot Confusion Matrix for All SNR Levels
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix (All SNR Levels)")
        axes[0, 0].set_xlabel("Predicted Label")
        axes[0, 0].set_ylabel("True Label")

        # Plot Confusion Matrix for SNR > 5 dB
        if conf_matrix_snr_above_5 is not None:
            sns.heatmap(conf_matrix_snr_above_5, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0, 1])
            axes[0, 1].set_title("Confusion Matrix (SNR > 5 dB)")
            axes[0, 1].set_xlabel("Predicted Label")
            axes[0, 1].set_ylabel("True Label")
        else:
            axes[0, 1].text(0.5, 0.5, "No Samples with SNR > 5 dB", 
                            ha='center', va='center', fontsize=12)
            axes[0, 1].set_title("Confusion Matrix (SNR > 5 dB)")

        # Plot Accuracy vs. SNR
        axes[1, 0].plot(unique_snrs, accuracy_per_snr, 'b-o', label='Recognition Accuracy')
        axes[1, 0].plot(peak_snr, peak_accuracy, 'ro')  # Mark peak accuracy
        axes[1, 0].text(peak_snr, peak_accuracy + 1, f"{peak_accuracy:.2f}%", 
                        ha='center', va='bottom', fontsize=10, 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        axes[1, 0].set_title("Recognition Accuracy vs. SNR")
        axes[1, 0].set_xlabel("SNR (dB)")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].grid(True)

        # Plot Accuracy vs. SNR per Modulation Type
        for mod, accuracies in modulation_traces:
            axes[1, 1].plot(unique_snrs, accuracies, '-o', label=mod)
        axes[1, 1].set_title("Accuracy vs. SNR per Modulation Type")
        axes[1, 1].set_xlabel("SNR (dB)")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend(loc='upper left', fontsize=8)
        axes[1, 1].grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        output_file = os.path.join(common_vars.stats_dir, f"{self.get_model_name()}_analysis.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")

        if show_plot:
            plt.show()

        # Return statistics
        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_over_5dB": accuracy_over_5dB,
            "accuracy_per_snr": dict(zip(unique_snrs, accuracy_per_snr)),
            "best_accuracy": peak_accuracy / 100, # scale to 0-1.0
            "peak_snr": peak_snr,
        }