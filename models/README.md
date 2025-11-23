# Model Zoo

This directory contains the trained models used in the project *Wireless Signal Classification via Deep Learning*.  
Models span multiple datasets (RML2016.10a, RML2018.01A, DeepRadar2022) and architectures (LSTM, RNN-LSTM, CNN-BiLSTM, GAN-based).  
All models are saved in the TensorFlow `.keras` format.

## Contents

| File                            | Dataset            | Architecture              | Description |
|---------------------------------|--------------------|----------------------------|-------------|
| `rnn_lstm_w_SNR_5_2_1.keras`    | RML2016.10a        | RNN-LSTM (SNR-augmented)  | Main model with SNR embedding appended to I/Q time series. |
| `lstm_rnn_2024.keras`           | RML2016.10a        | Stacked LSTM-RNN          | Updated baseline architecture (2024). |
| `deepradar2022_cnn_bilstm_final.keras` | DeepRadar2022 | CNN + BiLSTM              | Final radar classifier with 1D CNN front-end and BiLSTM. |
| `RMLGAN2_model_9270.keras`      | RML2016.10a        | CNN-GAN Hybrid            | Experimental GAN-assisted classifier. |

---

# Model Cards

Below are detailed model cards for each trained model.

---

## Model Card: rnn_lstm_w_SNR_5_2_1.keras

### Overview
A recurrent neural network with LSTM layers trained on the RML2016.10a dataset.  
Includes explicit SNR embedding appended to I/Q sequences.

### Architecture Summary
- Input: 128 time steps, 3 channels (I, Q, SNR)
- Core: Multi-layer LSTM network
- Output: 11-class softmax

### Intended Use
Signal detection and modulation classification research.

### Limitations
Reduced performance at very low SNR (< -6 dB).

---

## Model Card: lstm_rnn_2024.keras

### Overview
Stacked LSTM-RNN model for modulation classification.  
Lightweight baseline for comparison experiments.

### Architecture Summary
- Input: 128 time steps, 2 channels (I, Q)
- Simplified LSTM stack
- Output: 11-class classifier

### Intended Use
Benchmarking and ablation studies.

### Limitations
Lower accuracy than SNR-augmented model.

---

## Model Card: deepradar2022_cnn_bilstm_final.keras

### Overview
Hybrid CNN + BiLSTM architecture trained on DeepRadar2022 dataset.

### Architecture Summary
- Input: 1024-length radar sweeps
- Feature extractor: 1D CNN
- Temporal modeling: BiLSTM
- Output: Radar class labels

### Intended Use
Radar waveform classification.

### Limitations
Large memory footprint; requires strong GPU hardware.

---

## Model Card: RMLGAN2_model_9270.keras

### Overview
An experimental GAN-assisted classifier designed for augmentation experiments.

### Architecture Summary
- CNN-based encoder
- GAN components incorporated during training
- Output: RML2016.10a modulation classes

### Intended Use
Research on synthetic IQ augmentation.

### Limitations
Not used as a final production model; experimental accuracy.

---

# Usage

Load any model:

```
from tensorflow.keras.models import load_model
model = load_model("models/<filename>.keras")
```

---

# Notes

- All models follow the `.keras` format naming convention.
- Preprocessing pipelines are defined under `src/ml_wireless_classification/`.
- For radar models, GPU memory >= 12 GB is recommended.
