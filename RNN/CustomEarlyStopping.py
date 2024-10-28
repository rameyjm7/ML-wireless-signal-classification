
from tensorflow.keras.callbacks import EarlyStopping, Callback

class CustomEarlyStopping(Callback):
    def __init__(self, monitor="val_accuracy", min_delta=0.01, patience=5, restore_best_weights=True):
        """
        Custom early stopping to stop training if validation accuracy exceeds the current highest by min_delta.

        Parameters:
        - monitor (str): Metric to monitor (default is 'val_accuracy').
        - min_delta (float): Minimum improvement over best accuracy to continue training.
        - patience (int): Number of epochs to wait after last improvement.
        - restore_best_weights (bool): Whether to restore the weights of the best epoch.
        """
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best = -float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            print(f"Warning: Metric {self.monitor} is not available.")
            return
        
        # If current accuracy exceeds the best by min_delta, update best and reset wait counter
        if current > self.best + self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # Increment the wait counter if no improvement
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print("Restoring model weights from the best epoch.")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")