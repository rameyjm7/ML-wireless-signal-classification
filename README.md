# Wireless Signal Classification via Deep Learning

**Authors:** Jacob M. Ramey, Paras Goda

---

# Badges
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![Docker](https://img.shields.io/badge/Docker-GPU--Ready-blue)

---

# Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Results: RML2016](#results-rml2016)
- [Results: RML2018](#results-rml2018)
- [Results: DeepRadar2022](#results-deepradar2022)
- [More to Come](#more-to-come)
- [Installation](#installation)
- [Citation](#citation)

---

# Overview

This repository contains the core training pipelines, evaluation scripts, models, and documentation for a set of deep‑learning–based modulation classification experiments.  
The work combines practical engineering workflows (Docker, GPU training, reproducible pipelines) with structured experimentation across multiple wireless datasets.

The implemented models span:
- LSTM and BiLSTM architectures for raw I/Q modeling  
- CNN front‑ends for frequency‑domain structure  
- Dual‑head time–frequency fusion models  
- AC‑GAN augmentation for SNR/class balancing  
- Early transformer‑based spectral modeling  

The code supports large‑scale training (up to ~2M samples) and portable GPU environments via Docker and Apptainer.

---

# Repository Structure

```
src/ml_wireless_classification/   Core training, preprocessing, OSIL, GAN modules
notebooks/                        Reproducible training notebooks
docker/                           CUDA-enabled Docker + Apptainer environment
models/                           Trained .keras models
docs/                             Technical reports and supporting materials
```

---

# Datasets

### RML2016.10A  
11 modulation types, SNR from −20 dB to +18 dB.

### RML2018.01A  
24 classes; used for cross‑dataset generalization and larger‑scale evaluation.

### DeepRadar2022  
Radar waveform dataset used for CNN–BiLSTM hybrid models.

---

# Results: RML2016

### Summary
- Accuracy (all SNR): 67.8%  
- Accuracy (SNR > 5 dB): 94%  
- Macro F1: 0.68  
- Weighted F1: 0.68  

### Confusion Matrix
![RML2016 Confusion Matrix](https://github.com/user-attachments/assets/6eebbb20-105d-4c9c-ba17-7f2ec11e070f)

---

# Results: RML2018

### Overall Accuracy
Approx. 72% across 72,000 evaluation samples.

### Results Figure 1
![RML2018 Image 1](https://github.com/user-attachments/assets/e23bbd81-9f4f-4d7a-9bd4-11e0a3625044)

### Classification Report (All SNRs)

```
precision    recall  f1-score   support

128APSK 0.36 0.16 0.23 3000
128QAM  0.42 0.49 0.45 3000
16APSK  0.91 0.90 0.90 3000
16PSK   0.86 0.76 0.81 3000
16QAM   0.75 0.92 0.83 3000
256QAM  0.92 0.74 0.82 3000
32APSK  0.87 0.80 0.83 3000
32PSK   0.98 0.97 0.98 3000
32QAM   0.91 0.90 0.90 3000
4ASK    0.63 0.56 0.60 3000
64APSK  0.45 0.69 0.54 3000
64QAM   0.60 0.84 0.70 3000
8ASK    0.46 0.83 0.60 3000
8PSK    0.63 0.90 0.74 3000
AM-DSB-SC 0.39 0.15 0.22 3000
AM-DSB-WC 1.00 1.00 1.00 3000
AM-SSB-SC 0.66 0.60 0.63 3000
AM-SSB-WC 0.71 0.45 0.55 3000
BPSK    0.78 0.85 0.81 3000
FM      0.94 0.98 0.96 3000
GMSK    0.88 0.87 0.87 3000
OOK     0.84 0.96 0.90 3000
OQPSK   0.29 0.04 0.06 3000
QPSK    0.78 0.94 0.85 3000

accuracy 0.72 72000
```

### Additional RML2018 Figures

![RML2018 Image 2](https://github.com/user-attachments/assets/99d3b667-93d4-430e-abf3-aa6b4c743a31)
![RML2018 Image 3](https://github.com/user-attachments/assets/5e8d2c18-5c62-489e-84ea-8b2648eca610)

---

# Results: DeepRadar2022

### CNN–BiLSTM Hybrid Evaluation

![DeepRadar Image 1](https://github.com/user-attachments/assets/a2e6d2dc-ef18-4bdd-a8c8-31d2bbb77a0f)
![DeepRadar Image 2](https://github.com/user-attachments/assets/68843377-7fc4-45ba-9f16-9a40b9ecc2c9)
![DeepRadar Image 3](https://github.com/user-attachments/assets/0754f4da-e8d6-4cd0-9627-056e932a2865)
![DeepRadar Image 4](https://github.com/user-attachments/assets/c9eb6c5b-737f-4273-ba68-0ac3d13e3aab)

---

# More to Come

Future additions planned:

- Expanded DeepRadar model cards  
- Transformer-based spectral/temporal modeling  
- Cross-dataset robustness tests  
- Real-time inference deployment on edge hardware  

---

# Installation

### Docker (GPU)

```
cd docker
make build
make run
```

### Local Virtual Environment

```
python3 -m venv ~/python
source ~/python/bin/activate
pip install -e .
```

### HPC / Apptainer

```
module load apptainer
make asif
apptainer run --nv ml-wireless-signal-classification-hpc.sif jupyter lab --no-browser
```

---

# Citation

Ramey, J. M., and Goda, P. (2025). Wireless Signal Classification via Deep Learning.  
https://github.com/rameyjm7/ML-wireless-signal-classification
