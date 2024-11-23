import os
import ctypes
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
import gc
import numpy as np
from scipy.fft import fft
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import (
    BaseModulationClassifier,
)
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping

from ml_wireless_classification.base.CommonVars import common_vars, RUN_MODE
from ml_wireless_classification.base.TestingUtils import convert_and_clean_data
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", 
                 stats_path="model_stats.json",
                 model_name = "rnn_lstm_w_SNR_5_2_1_ensemble"):
        super().__init__(data_path, model_path, stats_path)
        self.learning_rate = 0.0001
        self.name = "rnn_lstm_w_SNR"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(script_dir, "..", "..", "RML2016.10a_dict.pkl")
        
        self.model_name = model_name
        self.model_path = os.path.join(script_dir, "models", f"{self.model_name}.keras")

    def am_dsb_demod(self, iq_signal):
        analytic_signal = hilbert(iq_signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def wbfm_demod(self, iq_signal, sample_rate=1.0):
        phase = np.unwrap(np.angle(iq_signal[:, 0] + 1j * iq_signal[:, 1]))
        demodulated = np.diff(phase) * (sample_rate / (2 * np.pi))
        return np.pad(demodulated, (1, 0), mode="constant")
    
    def prepare_data(self):
        X_existing, X_combined_branch, y = [], [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T  # Shape: (128, 2)
                snr_signal = np.full((128, 1), snr)  # Shape: (128, 1)

                # Add SNR to the first branch
                combined_signal = np.hstack([iq_signal, snr_signal])  # Shape: (128, 3)

                # Perform AM-DSB and WBFM demodulation
                am_dsb_feature_i = self.am_dsb_demod(np.abs(signal[0]))  # Shape: (128,)
                am_dsb_feature_q = self.am_dsb_demod(np.abs(signal[1]))  # Shape: (128,)
                wbfm_feature = self.wbfm_demod(iq_signal)  # Shape: (128,)

                # Expand dimensions for consistency
                am_dsb_feature_i = np.expand_dims(am_dsb_feature_i, axis=-1)  # Shape: (128, 1)
                am_dsb_feature_q = np.expand_dims(am_dsb_feature_q, axis=-1)  # Shape: (128, 1)
                wbfm_feature = np.expand_dims(wbfm_feature, axis=-1)  # Shape: (128, 1)
                
                # print("am_dsb_feature_i shape:", am_dsb_feature_i.shape)
                # print("am_dsb_feature_q shape:", am_dsb_feature_q.shape)
                # print("wbfm_feature shape:", wbfm_feature.shape)

                # Combine AM-DSB, WBFM, and SNR features for the second branch
                combined_branch_features = np.hstack([am_dsb_feature_i, am_dsb_feature_q, wbfm_feature])  # Shape: (128, 3)
                # print("Combined shape (expected):", combined_branch_features.shape)

                # Collect data
                X_existing.append(combined_signal)  # Shape: (128, 3)
                X_combined_branch.append(combined_branch_features)  # Shape: (128, 3)
                y.append(mod_type)

        # Convert to numpy arrays
        X_existing = np.array(X_existing)  # Shape: (num_samples, 128, 3)
        X_combined_branch = np.array(X_combined_branch)  # Shape: (num_samples, 128, 3)
        y = np.array(y)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split into training and test sets
        X_existing_train, X_existing_test, X_combined_train, X_combined_test, y_train, y_test = train_test_split(
            X_existing, X_combined_branch, y_encoded, test_size=0.2, random_state=42
        )

        return (
            (X_existing_train, X_combined_train, y_train),
            (X_existing_test, X_combined_test, y_test),
        )

    def evaluate(self, X_test, y_test):
        # Unpack the inputs for the two branches
        X_existing_test, X_combined_test = X_test

        # Evaluate overall test accuracy
        test_loss, test_acc = self.model.evaluate([X_existing_test, X_combined_test], y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        
        # Generate predictions
        y_pred = self.model.predict([X_existing_test, X_combined_test])
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

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            pretrained_model_name = "rnn_lstm_w_SNR_5_2_1"
            self.pretrained_model_path = os.path.join(script_dir, "models", f"{pretrained_model_name}.keras")
            self.ensemble_model_path = os.path.join(script_dir, "models", f"{pretrained_model_name}_ensemble.keras")
            
            if not os.path.exists(self.pretrained_model_path):
                raise FileNotFoundError(f"Pre-trained model not found at {self.pretrained_model_path}")
            
            # Load and freeze the pre-trained model
            print(f"Loading pre-trained model from {self.pretrained_model_path}")
            pretrained_model = load_model(self.pretrained_model_path)
            pretrained_model.trainable = False

            # First branch: Pre-trained model
            existing_model_output = pretrained_model.output  # Shape: (batch_size, timesteps, features)
            existing_input = pretrained_model.input

            # Second branch: AM-DSB, WBFM, and SNR features
            combined_branch_input = Input(shape=(128, 3), name="combined_branch_input")  # Shape: (batch_size, 128, 3)
            lstm_features = Bidirectional(LSTM(128, return_sequences=False, name="lstm_combined_features"))(combined_branch_input)
            dense_features = Dense(128, activation="relu", name="dense_combined_features")(lstm_features)

            # Combine both branches
            combined = Concatenate(name="combine_existing_new_features")([existing_model_output, dense_features])
            x = Dense(128, activation="relu", name="dense_combined")(combined)
            x = Dropout(0.1, name="dropout_combined")(x)
            output = Dense(num_classes, activation="softmax", name="final_output")(x)

            # Define the ensemble model
            self.model = Model(inputs=[existing_input, combined_branch_input], outputs=output)

            # Compile the model
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )

    def save_model(self):
        print(f"Saving ensemble model to {self.model_path}")
        self.model.save(self.model_path)

    def setup(self):
        self.load_data()
        train_data, test_data = self.prepare_data()
        
        # Unpack training and testing data
        X_train, X_combined_train, y_train = train_data
        X_test, X_combined_test, y_test = test_data
        
        # Apply class weighting for imbalanced datasets
        self.apply_class_weighting(y_train)

        # Define the input shape based on the second branch's data (combined features)
        input_shape = (X_combined_train.shape[1], X_combined_train.shape[2])  # Features per timestep
        num_classes = len(np.unique(y_train))  # Number of unique classes in labels
        self.build_model(input_shape, num_classes)

        # Return the structured data for training and testing
        return (X_train, X_combined_train, y_train), (X_test, X_combined_test, y_test)

    def main(self, mode : RUN_MODE):
        train_data, test_data = self.setup()
        X_train, X_combined_train, y_train = train_data
        X_test, X_combined_test, y_test = test_data

        if mode == RUN_MODE.TRAIN_CONTINUOUSLY:
            self.train_continuously(
                [X_train, X_combined_train],
                y_train,
                [X_test, X_combined_test],
                y_test,
                batch_size=64,
                use_clr=True,
                clr_step_size=10,
            )
        elif mode == RUN_MODE.TRAIN:
            self.train(
                [X_train, X_combined_train],
                y_train,
                [X_test, X_combined_test],
                y_test,
                batch_size=64,
                use_clr=True,
                clr_step_size=10,
            )
        elif mode == RUN_MODE.EVALUATE_ONLY:
            print("Evaluating only...")
            
        self.evaluate([X_test, X_combined_test], y_test)
        predictions = self.predict([X_test, X_combined_test])
        print("Predicted Labels: ", predictions[:5])
        print("True Labels: ", self.label_encoder.inverse_transform(y_test[:5]))
        
        stats_dict = self.plot_all_analysis(X_test, X_combined_test, y_test, self.label_encoder, model_name, False)
        self.update_and_save_stats(stats_dict)

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
        """
        Trains the model and updates statistics.

        Parameters:
        - X_train: Tuple containing inputs for both branches of the model during training.
        - y_train: Training labels.
        - X_test: Tuple containing inputs for both branches of the model during validation.
        - y_test: Validation labels.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        - use_clr (bool): Whether to use Cyclical Learning Rate (CLR) scheduler.
        - clr_step_size (int): Step size for CLR scheduler.

        Returns:
        - history: Training history object from Keras.
        """
        callbacks = []

        # Set up Cyclical Learning Rate scheduler if enabled
        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        # Train the model
        history = self.model.fit(
            [X_train[0], X_train[1]],  # First branch: pre-trained input, second branch: combined features
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_test[0], X_test[1]], y_test),  # Validation inputs
            callbacks=callbacks,
            class_weight=self.class_weights_dict,
        )

        # Get current and best validation accuracies
        current_accuracy = max(history.history["val_accuracy"])
        best_accuracy = self.stats.get("best_accuracy", 0)

        # Update the stats dictionary
        new_stats = {
            "current_accuracy": current_accuracy,
            "best_accuracy": max(current_accuracy, best_accuracy),
            "epochs_ran": self.stats.get("epochs_ran", 0) + epochs,
            "last_run_epochs": epochs,
        }

        # Update stats and save the model if needed
        self.update_and_save_stats(new_stats)

        return history

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
    
    def plot_all_analysis(self, X_test, X_combined_test, y_test, label_encoder, model_name, show_plot=False):
        """
        Combines all analysis and visualization into a single function.
        Organizes the plots in a single figure with a dynamic grid layout.
        Saves the figure to the specified directory with the model name as the file name.
        Returns a dictionary of statistics such as accuracy over 5 dB and accuracy per SNR.
        """
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        common_vars.stats_dir = os.path.join(script_dir, "stats")

        model = self.model
        # --- Confusion Matrix for All SNR Levels ---
        y_pred = np.argmax(model.predict([X_test, X_combined_test], verbose=False), axis=1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        overall_accuracy = accuracy_score(y_test, y_pred) * 100

        # --- Confusion Matrix for SNR > 5 dB ---
        snr_above_5_indices = np.where(X_test[:, :, 2].mean(axis=1) > 5)
        X_test_snr_above_5 = X_test[snr_above_5_indices]
        X_combined_test_snr_above_5 = X_combined_test[snr_above_5_indices]
        y_test_snr_above_5 = y_test[snr_above_5_indices]

        if len(X_test_snr_above_5) > 0:
            y_pred_snr_above_5 = np.argmax(
                model.predict([X_test_snr_above_5, X_combined_test_snr_above_5], verbose=False), axis=1
            )
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
            X_combined_snr = X_combined_test[snr_indices]
            y_snr = y_test[snr_indices]
            if len(y_snr) > 0:
                y_pred_snr = np.argmax(model.predict([X_snr, X_combined_snr], verbose=0), axis=1)
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
                X_combined_mod_snr = X_combined_test[mod_snr_indices]
                y_mod_snr = y_test[mod_snr_indices]
                if len(y_mod_snr) > 0:
                    y_pred_mod_snr = np.argmax(model.predict([X_mod_snr, X_combined_mod_snr], verbose=False), axis=1)
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
        output_file = os.path.join(common_vars.stats_dir, f"{model_name}_analysis.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")

        if show_plot:
            plt.show()

        # Return statistics
        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_over_5dB": accuracy_over_5dB,
            "accuracy_per_snr": dict(zip(unique_snrs, accuracy_per_snr)),
            "peak_accuracy": peak_accuracy,
            "peak_snr": peak_snr,
        }

        

if __name__ == "__main__":
    # set the model name
    model_name = "rnn_lstm_w_SNR_5_2_1_ensemble"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory

    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.models_dir = os.path.join(script_dir, " models")
    model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

    # Usage Example
    print("Data path:", data_path)
    print("Model path:", model_path) 
    print("Stats path:", stats_path)

    # Initialize the classifier
    classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
    classifier.main(mode=RUN_MODE.EVALUATE_ONLY)
