# Documentation

This directory contains the formal project reports and supporting research documents for the work:

Wireless Signal Classification via Deep Learning  
Jacob M. Ramey, Paras Goda (Equal Contributors)

Primary Report: Wireless Signal Classification via Deep Learning (Generative Adversarial, Open Set Incremental and Dual Head)

## Overview

The documents in this directory detail the full academic analysis, experimental results, and architectural designs for the wireless-signal classification research project. The work includes:

- Cross-dataset evaluation (RML2016 → RML2018a)
- Open-set incremental learning (OSIL)
- GAN-assisted sample balancing
- Dual-head time–frequency neural architecture
- Transformer-based preliminary experiments

## File List

1. Wireless Signal Classification via Deep Learning – Final Jacob M. Ramey.pdf  
   Formal 7‑page technical report describing the OSIL engine, GAN augmentation, dual‑head CNN+BiLSTM, and cross-corpus experiments.

2. Wireless Signal Classification via Machine Learning – Final.pdf  
   Earlier version of the analysis focusing on baseline LSTM and CNN comparisons.

## Summary of Included Research

The main report presents:

### Cross‑Corpus Generalization
A bidirectional LSTM trained on RML2016 is ported unchanged to the 24‑class RML2018a dataset, achieving 74% accuracy after 200 epochs.

### Open‑Set Incremental Learning (OSIL)
A lightweight incremental-learning framework combining entropy-based gating, Mahalanobis rejection, replay buffers, and classifier expansion.

### GAN‑Assisted Augmentation
A class‑conditioned AC‑GAN generates high‑SNR synthetic waveforms to counter dataset imbalance.

### Dual‑Head Time–Frequency Network
A fusion model combining:
- Frequency‑domain CNN features from FFT magnitude patches.
- Time‑domain BiLSTM features from I/Q sequences.

### Transformer Exploration
Initial experiments with a Spectral Transformer using I/Q patches as tokens.

## Usage

These documents serve as the reference material for the experiments and architectures found in:

- notebooks/
- src/ml_wireless_classification/
- models/
- docker/

They describe the theoretical and experimental motivations underlying the models and pipelines implemented in this repository.

