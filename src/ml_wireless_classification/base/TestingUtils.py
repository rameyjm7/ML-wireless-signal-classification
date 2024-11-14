import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def convert_and_clean_data(data):
    """
    Convert complex data to real, replacing -inf with the minimum float32 value,
    +inf with the maximum float32 value, and NaN with 0.
    
    Parameters:
        data (np.ndarray): Complex or real-valued data array.
        
    Returns:
        np.ndarray: Cleaned real-valued data array.
    """
    # Convert complex to real by taking only the real part
    data_real = np.real(data)
    
    # Replace -inf, +inf, and NaN values
    data_real = np.where(np.isinf(data_real) & (data_real < 0), np.finfo(np.float32).min, data_real)
    data_real = np.where(np.isinf(data_real) & (data_real > 0), np.finfo(np.float32).max, data_real)
    data_real = np.where(np.isnan(data_real), 0, data_real)
    
    return data_real


def clean_training_data(X, y):
    """
    Cleans X and y by ensuring that all elements are scalars and removing infinities.
    Prints a message if a sequence (list, tuple, array) or non-numeric value is found.
    """
    def check_and_clean_array(arr, array_name):
        cleaned_arr = []
        for i, row in enumerate(arr):
            cleaned_row = []
            for j, value in enumerate(row):
                # Check if value is scalar
                if np.isscalar(value):
                    # Check for infinities or non-finite values
                    if np.isinf(value) or np.isnan(value) or value > np.finfo(np.float32).max:
                        # print(f"Warning: {array_name}[{i}][{j}] has an infinity or too large value. Setting to 0.")
                        cleaned_row.append(0)  # Replace infinities or overly large values with 0
                    else:
                        cleaned_row.append(value)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    # If it's a sequence, take the first element as a workaround (optional)
                    sub_value = value[0] if len(value) > 0 else 0
                    if np.isinf(sub_value) or np.isnan(sub_value) or sub_value > np.finfo(np.float32).max:
                        # print(f"Warning: {array_name}[{i}][{j}] contains infinity or too large in sequence. Setting to 0.")
                        sub_value = 0
                    cleaned_row.append(sub_value)
                    # print(f"Warning: {array_name}[{i}][{j}] is a sequence. Taking the first element.")
                elif isinstance(value, str):
                    # Handle string values with a warning
                    # print(f"Warning: {array_name}[{i}][{j}] is a string. Removing and setting to 0.")
                    cleaned_row.append(0)
                else:
                    # print(f"Warning: Unexpected data type at {array_name}[{i}][{j}]: {type(value)}")
                    cleaned_row.append(0)  # Default to 0 if type is unexpected
            cleaned_arr.append(cleaned_row)
        
        # Ensure cleaned_arr is a 2D array of fixed-length rows
        max_length = max(len(row) for row in cleaned_arr)
        # Pad rows with zeros if they are shorter than max_length
        cleaned_arr = [row + [0] * (max_length - len(row)) for row in cleaned_arr]
        
        return np.array(cleaned_arr, dtype=float)
    
    # Clean X and y
    X_cleaned = check_and_clean_array(X, "X_train")
    y_cleaned = np.array([elem if np.isscalar(elem) else elem[0] for elem in y], dtype=float)

    return X_cleaned, y_cleaned

def ensure_2d(arr, name):
    """
    Ensures the array is 2D by reshaping if necessary.
    """
    if arr.ndim == 1:
        print(f"Warning: {name} is 1-dimensional. Reshaping to 2D.")
        arr = arr.reshape(-1, 1)
    return arr



def plot_feature_importance(clf : RandomForestClassifier, feature_dict : dict, X_test, y_test):
    # Feature importance for the classifier
    feature_names = list(feature_dict.keys())
    importances = clf.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    # Plot sorted feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance for Modulation Classification")
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance at the top
    plt.show()
    
def plot_confusion_matrix(model, X_test, y_test, label_encoder):
    # Confusion matrix for overall test set
    y_pred_test = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Multi-Class Modulation Classification")
    plt.show()

    # Print Classification Report
    print("Classification Report for Modulation Types:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))


def plot_accuracy_per_snr(model : RandomForestClassifier, X_test, y_test):
    # Evaluate accuracy for each SNR level
    unique_snrs = sorted(set(X_test[:, -1]))  # Get unique SNR levels from test set
    accuracy_per_snr = []

    for snr in unique_snrs:
        # Select samples with the current SNR
        snr_indices = np.where(X_test[:, -1] == snr)
        X_snr = X_test[snr_indices]
        y_snr = y_test[snr_indices]

        # Predict and calculate accuracy
        y_pred = model.predict(X_snr)
        accuracy = accuracy_score(y_snr, y_pred)
        accuracy_per_snr.append(accuracy * 100)  # Convert to percentage

        print(f"SNR: {snr} dB, Accuracy: {accuracy * 100:.2f}%")

    # Find the peak accuracy and its corresponding SNR
    peak_accuracy = max(accuracy_per_snr)
    peak_snr = unique_snrs[accuracy_per_snr.index(peak_accuracy)]

    # Plot Recognition Accuracy vs. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, accuracy_per_snr, 'b-o', label='Recognition Accuracy')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Recognition Accuracy (%)")
    plt.title(f"Recognition Accuracy vs. SNR (Peak Accuracy: {peak_accuracy:.2f}%)")

    # Mark the peak accuracy point
    plt.plot(peak_snr, peak_accuracy, 'ro')  # Red dot at the peak
    plt.text(peak_snr, peak_accuracy + 1, f"{peak_accuracy:.2f}%", 
            ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.show()
    

def plot_accuracy_v_snr_per_classification(model : RandomForestClassifier, X_test, y_test, label_encoder):
    # Get unique modulation types using label encoder classes
    unique_modulations = label_encoder.classes_  # This will be the actual class names
    unique_snrs = sorted(set(X_test[:, -1]))  # Get unique SNR levels from test set

    # Initialize a list to store modulation, accuracy per SNR, and peak accuracy for sorting
    modulation_traces = []

    # Calculate accuracy for each modulation type and SNR level
    for mod_index, mod in enumerate(unique_modulations):
        accuracies = []
        for snr in unique_snrs:
            # Select samples with the current modulation type and SNR
            mod_snr_indices = np.where((y_test == mod_index) & (X_test[:, -1] == snr))
            X_mod_snr = X_test[mod_snr_indices]
            y_mod_snr = y_test[mod_snr_indices]

            # Predict and calculate accuracy
            if len(y_mod_snr) > 0:  # Check if there are samples for this SNR and modulation type
                y_pred = model.predict(X_mod_snr)
                accuracy = accuracy_score(y_mod_snr, y_pred) * 100  # Convert to percentage
            else:
                accuracy = np.nan  # No data for this SNR-modulation combination

            accuracies.append(accuracy)

        # Calculate peak accuracy for this modulation type
        valid_accuracies = [acc for acc in accuracies if not np.isnan(acc)]
        peak_accuracy = max(valid_accuracies) if valid_accuracies else 0
        peak_snr = unique_snrs[accuracies.index(peak_accuracy)] if peak_accuracy > 0 else None

        # Store the modulation trace data along with peak accuracy for sorting
        modulation_traces.append((mod, accuracies, peak_accuracy, peak_snr))

    # Sort the modulation types by peak accuracy in descending order
    modulation_traces = sorted(modulation_traces, key=lambda x: x[2], reverse=True)

    # Plot Recognition Accuracy vs. SNR for each modulation type
    plt.figure(figsize=(12, 8))
    for mod, accuracies, peak_accuracy, peak_snr in modulation_traces:
        # Plot the trace for the modulation type
        label = f'{mod} (Peak: {peak_accuracy:.2f}% at {peak_snr} dB)' if peak_accuracy > 0 else mod
        plt.plot(unique_snrs, accuracies, '-o', label=label)

        # Mark the peak accuracy point if it exists
        if peak_accuracy > 0 and peak_snr is not None:
            plt.plot(peak_snr, peak_accuracy, 'ro')  # Red dot at the peak
            plt.text(peak_snr, peak_accuracy + 1, f"{peak_accuracy:.2f}%", 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.xlabel("SNR (dB)")
    plt.ylabel("Recognition Accuracy (%)")
    plt.title("Recognition Accuracy vs. SNR per Modulation Type")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 110)
    plt.xlim(-20,20)
    plt.show()
