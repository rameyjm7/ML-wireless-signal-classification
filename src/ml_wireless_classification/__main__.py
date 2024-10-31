import faulthandler
faulthandler.enable()

import os
import numpy as np
from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.rnn_lstm_w_SNR import ModulationLSTMClassifier

def main(model_name, data_path = None):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths with the script directory as the base
    if not data_path:
        data_path = os.path.join(script_dir, "..", "..", "..", "RML2016.10a_dict.pkl")  # One level up from the script's directory
    else:
        # allow updating path
        print(f"data path set to {data_path})")
        
        
    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.models_dir = os.path.join(script_dir, "models")
    model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

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

    # Build the model (load if it exists)
    input_shape = (X_train.shape[1], X_train.shape[2])  # Time steps and features (I, Q, SNR, BW)
    num_classes = len(np.unique(y_train))  # Number of unique modulation types
    classifier.build_model(input_shape, num_classes)

    # Train continuously with cyclical learning rates
    classifier.train_continuously(
        X_train, y_train, X_test, y_test, batch_size=64, use_clr=True, clr_step_size=10
    )

    # Evaluate the model
    classifier.evaluate(X_test, y_test)

    # Optional: Make predictions on the test set
    predictions = classifier.predict(X_test)
    print("Predicted Labels: ", predictions[:5])
    print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))

if __name__ == "__main__":
    # set the model name 
    RML2016_data_pkl_path = "/home/dev/workspace/ML-wireless-signal-classification/RNN/src/ml_wireless_classification/../RML2016.10a_dict.pkl"
    model_name = "rnn_lstm_w_SNR_5_2_1"
    main(model_name)
