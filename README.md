# Wireless Signal Classification via Deep Learning (RNN-LSTM Models)

**Authors:** Jacob Ramey, Paras Goda  
**Repository:** [ML-wireless-signal-classification](https://github.com/rameyjm7/ML-wireless-signal-classification)

---

## Purpose

This repository contains the code and experiments for the research paper:  
**Wireless Signal Classification via Deep Learning (RNN-LSTM Models)**  
[Read the paper (PDF)](https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/Wireless%20Signal%20Classification%20via%20Machine%20Learning%20-%20Final.pdf)

The work focuses on classifying radio signals into one of 11 modulation types using deep learning methods, primarily the Recurrent Neural Network – Long Short-Term Memory (RNN–LSTM) architecture.  
Other traditional machine learning approaches were explored for comparison and validation, but the LSTM-based model achieved the highest accuracy. Full results and experimental details are presented in the accompanying paper.

---

## Datasets

The primary dataset used is [RADIOML 2016.10A](https://www.deepsig.ai/datasets/) by DeepSig.  
It contains complex I/Q samples representing 11 modulation types across SNR levels from -20 dB to +18 dB.

| Modulations | 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM |

---

## Repository Structure

```
src/ml_wireless_classification/   # Core package
tests/                            # Notebooks for model evaluation
docker/                           # Docker build files and usage guide
models/                           # Saved Keras models
```

---

## Model Overview

### RNN-LSTM Architecture

The RNN-LSTM model processes I/Q time series data and an appended SNR feature vector (128-bit extension).  
This structure allows the network to capture both temporal and amplitude-phase dynamics in the modulated signal.

**Model:** `RNN_LSTM_5_2_1.keras`  
**Notebook:** [tests/rnn_lstm.ipynb](https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/tests/rnn_lstm.ipynb)

---

## Classification Performance

### All SNR Levels
| Metric | Value |
|:-------|------:|
| Accuracy | 67.8 % |
| Macro F1 | 0.68 |
| Weighted F1 | 0.68 |
| Samples | 44,000 |

### SNR > 5 dB
| Metric | Value |
|:-------|------:|
| Accuracy | 94.0 % |
| Macro F1 | 0.93 |
| Weighted F1 | 0.93 |
| Samples | 15,332 |

![Confusion Matrix – All SNRs](https://github.com/user-attachments/assets/6eebbb20-105d-4c9c-ba17-7f2ec11e070f)

---

## Installation and Usage

### Option 1 – Docker

Refer to the [Docker README](https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/docker/README.md)  
for GPU-enabled build and execution instructions.

### Option 2 – Local Virtual Environment

```bash
apt install python3-venv
python3 -m venv ~/python
source ~/python/bin/activate
pip install -e .
python -m ml_wireless_classification
```

The application will train and save the model automatically under `src/ml_wireless_classification/models/`.

---

## GPU Environment

- TensorFlow 2.12.0  
- Python 3.10  
- cuDNN 8.6  
- CUDA 11.8  

---

## Future Work

- Incorporate DeepRadar2022 and other large-scale signal datasets  
- Extend LSTM network depth and explore Bidirectional architectures  
- Integrate phase and frequency-domain augmentations  
- Deploy real-time inference on Jetson hardware

---

## Citation

If using this repository, please cite:

> Ramey, J. M., & Goda, P. (2025). Wireless Signal Classification via Deep Learning (RNN-LSTM Models).  
> GitHub Repository: https://github.com/rameyjm7/ML-wireless-signal-classification

---
