import faulthandler
import os
import numpy as np
from ml_wireless_classification.base.CommonVars import common_vars, RUN_MODE
from ml_wireless_classification.rnn_lstm_w_SNR import ModulationLSTMClassifier

# Enable fault handler for better debugging
faulthandler.enable()

def is_docker_environment():
    """
    Check if the script is running inside a Docker container by verifying the existence of /workspace/code.
    """
    return os.path.exists("/workspace/code")

if __name__ == "__main__":
    # Set the model name
    model_name = "rnn_lstm_w_SNR"

    # Determine base directory
    if is_docker_environment():
        base_dir = "/workspace/code/src/ml_wireless_classification"
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    data_path = os.path.join(base_dir, "..", "..", "RML2016.10a_dict.pkl")
    common_vars.stats_dir = os.path.join(base_dir, "stats")
    common_vars.models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(base_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(base_dir, "stats", f"{model_name}_stats.json")

    # Usage Example
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Stats path:", stats_path)

    mode = RUN_MODE.EVALUATE_ONLY
    # Initialize the classifier
    classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
    classifier.main(mode)
