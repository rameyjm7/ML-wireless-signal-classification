import faulthandler

faulthandler.enable()

import os
import numpy as np
from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.GenericModulationClassifier import GenericModulationClassifier


def main(model_name, data_path=None, make_new_dropout_model=False):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths with the script directory as the base
    if not data_path:
        data_path = os.path.join(
            script_dir, "..", "..", "RML2016.10a_dict.pkl"
        )  # One level up from the script's directory
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
    classifier = GenericModulationClassifier(model_name, data_path, model_path, stats_path).classifier

    # Load the dataset
    classifier.load_data()

    # Prepare the data
    X_train, X_test, y_train, y_test = classifier.prepare_data()

    # Build the model (load if it exists)
    input_shape = (
        X_train.shape[1],
        X_train.shape[2],
    )  # Time steps and features (I, Q, SNR, BW)
    num_classes = len(np.unique(y_train))  # Number of unique modulation types
    classifier.build_model(input_shape, num_classes)

    if make_new_dropout_model:
        dropouts = [0.5, 0.2, 0.1]
        # make a new model with the same weights, but different dropout rates
        new_model_name = (
            "rnn_lstm_w_SNR_"
            + str(dropouts[0])
            + "_"
            + str(dropouts[1])
            + "_"
            + str(dropouts[2])
        )
        new_model_path = os.path.join(script_dir, "models", f"{new_model_name}.keras")
        classifier.transfer_model_with_dropout_adjustment(
            classifier.model, new_model_path
        )
        return 0

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
    models = [
        "rnn_lstm_w_SNR",
        "rnn_lstm_w_SNR_5_2_1",
        "rnn_lstm_multifeature_generic",   # this model needs to be regenerated
        "ConvLSTM_FFT_Power_SNR",
        "ConvLSTM_IQ_SNR_k7_k3",
        "ConvLSTM_IQ_SNR_k7",
    ]
    model_name = models[2]
    main(model_name, None, False)
