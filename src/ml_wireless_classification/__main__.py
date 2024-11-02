import faulthandler

faulthandler.enable()

import os
import numpy as np
from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.GenericModulationClassifier import GenericModulationClassifier

if __name__ == "__main__":
    # set the model name
    models = [
        "rnn_lstm_w_SNR",
        "rnn_lstm_multifeature_generic",   # this model needs to be regenerated
        "ConvLSTM_FFT_Power_SNR",
        "ConvLSTM_IQ_SNR",
    ]
    model_name = models[1]
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
    classifier = GenericModulationClassifier(model_name, data_path, model_path, stats_path).classifier
    classifier.main()
