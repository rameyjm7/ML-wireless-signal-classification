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


    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, use_clr=False, clr_step_size=10):
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
                try:
                    self.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=batch_size, use_clr=use_clr, clr_step_size=clr_step_size)
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
        return test_acc
    
    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return predicted_labels
