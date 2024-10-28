# Purpose

This is a repository to hold the code for our research paper: Wireless Signal Classification via Machine Learning.

# Datasets

Download the RADIOML 2016.10A dataset from this link https://www.deepsig.ai/datasets/

Extract the dataset and it will be loaded on startup

# What does the SW do?
After the application starts, it will create a model if one isn't present, train it, validate it, then save itself for next run

# GPU Information
To use with a GPU:
https://www.tensorflow.org/install/source#gpu

# Versions
I'm using these versions:

tensorflow-2.12.0
Python 3.10	
cuDNN 8.6
CUDA 11.8

# Model Statistics Summary
Confusion matrixes coming soon
This table summarizes statistics for all models found

| Model Name                                | Date Created        |   Epochs Trained |   Best Accuracy |   Current Accuracy | Last Trained        |
|:------------------------------------------|:--------------------|-----------------:|----------------:|-------------------:|:--------------------|
| rnn_lstm_multifeature_generic_5_2_1_stats | 2024-10-27 21:52:19 |              160 |        0.527705 |           0.520023 | 2024-10-27 23:02:44 |
| rnn_lstm_w_SNR_stats                      | 2024-10-24 14:33:20 |             1315 |        0.657091 |           0.654955 | 2024-10-26 15:51:42 |
| rnn_lstm_w_SNR_5_2_1_stats                | 2024-10-26 22:03:21 |             4800 |        0.675909 |           0.674455 | 2024-10-27 23:04:09 |
| rnn_lstm_multifeature_generic_stats       | 2024-10-24 20:49:13 |             3763 |        0.550136 |           0.538636 | 2024-10-27 21:42:37 |
| rnn_lstm_multifeature_generic_w_fft_stats | 2024-10-25 21:57:40 |              880 |        0.517455 |           0.515477 | 2024-10-26 11:55:08 |