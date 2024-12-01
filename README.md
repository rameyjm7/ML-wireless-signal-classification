# Purpose

This is a repository to hold the code for our research paper: Wireless Signal Classification via Machine Learning. See our (in progress) paper here
https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/Wireless%20Signal%20Classification%20via%20Machine%20Learning.pdf

We want to classify a signal as 1 of 11 modulation types using a machine learning model, such as RNN-LSTM or a Random Forest Classifier

# Datasets

Download the RADIOML 2016.10A dataset from this link https://www.deepsig.ai/datasets/

Extract the dataset and it will be loaded on startup

# What does the SW do?

Its a python module with different machine learning models, statistics, and generic code for training and evaluation. 

# To install the module and run the code:

**Option 1: Using Docker**

You can just browse to the docker folder and view the [README.md](https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/docker/README.md) there for instructions


**Option 2: Running it directly**

// Install python3 virtual env

apt install python3-venv

// install virtual env somewhere

python3 -m venv /home/eng/python

// source the environment

source /home/eng/python/bin/activate

// install the module from the base of the repository

pip3 install -e . 

// run the module

python3 -m ml_wireless_classification




After the application starts, it will create a model if one isn't present, train it, validate it, then save itself for next run

Look under the src/ml_wireless_classification/ folder for two models I'm experimenting with


# GPU Information
To use with a GPU:
https://www.tensorflow.org/install/source#gpu

# Versions
I'm using these versions:

tensorflow-2.12.0

Python 3.10	

cuDNN 8.6

CUDA 11.8


# Top model

Look in this notebook for the results of the top model using an RNN LSTM with 128 bit IQ and 128 bit SNR feature (1 bit extended to 128)
https://github.com/rameyjm7/ML-wireless-signal-classification/blob/main/tests/rnn_lstm.ipynb

Using RNN_LSTM_5_2_1.keras

![image](https://github.com/user-attachments/assets/6eebbb20-105d-4c9c-ba17-7f2ec11e070f)

Classification Report for Modulation Types:
              precision    recall  f1-score   support

        8PSK       0.77      0.60      0.68      3990
      AM-DSB       0.50      0.77      0.61      4086
      AM-SSB       0.51      0.80      0.62      4005
        BPSK       0.84      0.66      0.74      4018
       CPFSK       0.83      0.64      0.72      3995
        GFSK       0.77      0.68      0.73      3967
        PAM4       0.79      0.77      0.78      4004
       QAM16       0.77      0.75      0.76      3921
       QAM64       0.69      0.86      0.77      3955
        QPSK       0.74      0.60      0.66      4088
        WBFM       0.47      0.32      0.38      3971

    accuracy                           0.68     44000
   macro avg       0.70      0.68      0.68     44000
weighted avg       0.70      0.68      0.68     44000

Classification Report for Modulation Types (SNR > 5 dB):
              precision    recall  f1-score   support

        8PSK       1.00      0.99      0.99      1420
      AM-DSB       0.65      1.00      0.79      1425
      AM-SSB       0.96      1.00      0.98      1398
        BPSK       1.00      0.99      0.99      1402
       CPFSK       1.00      1.00      1.00      1409
        GFSK       1.00      1.00      1.00      1337
        PAM4       0.99      0.99      0.99      1372
       QAM16       0.95      0.97      0.96      1417
       QAM64       0.97      0.93      0.95      1354
        QPSK       0.99      0.99      0.99      1423
        WBFM       1.00      0.44      0.61      1375

    accuracy                           0.94     15332
   macro avg       0.95      0.94      0.93     15332
weighted avg       0.95      0.94      0.93     15332

Using **Random Foresting** and many features I was able to achieve 90% accuracy with minimal training (~15 seconds)

![image](https://github.com/user-attachments/assets/b2a0f6de-1090-42c5-800b-b56288c22324)
![image](https://github.com/user-attachments/assets/06309dc4-e6c4-4356-87d8-fead459fe1e0)

The Feature importance is shown here

![image](https://github.com/user-attachments/assets/0cb5e8a3-e244-4918-9d82-caa986ecbe16)


In general, the most confusion was between WBFM and AM-DSB, and then followed by QAM64 and QAM16. To improve the model, features were targeted to improve the recognition here. 

# Model Statistics Summary

This table summarizes statistics for all models found, sorted by best accuracy.

| Model Name                                          | Date Created        |   Epochs Trained |   Best Accuracy |   Current Accuracy | Last Trained        |
|:----------------------------------------------------|:--------------------|-----------------:|----------------:|-------------------:|:--------------------|
| rnn_lstm_w_SNR_stats                                | 2024-10-24 14:33:20 |             2215 |        0.678091 |           0.670545 | 2024-11-07 12:20:06 |
| rnn_lstm_w_SNR_5_2_1_stats                          | 2024-10-26 22:03:21 |            13960 |        0.677909 |           0.677455 | 2024-11-01 10:21:04 |
| EnhancedModulationClassifier_KitchenSink_v2_stats   | 2024-11-05 13:18:16 |              920 |        0.554318 |           0.554205 | 2024-11-06 11:53:17 |
| EnhancedModulationClassifier_AMvsPSK_v2_2_2_2_stats | 2024-11-03 19:35:50 |             1140 |        0.550773 |           0.550136 | 2024-11-04 10:14:46 |
| rnn_lstm_multifeature_generic_stats                 | 2024-10-24 20:49:13 |             3803 |        0.550136 |           0.533341 | 2024-11-01 14:11:35 |
| EnhancedModulationClassifier_AMvsPSK_2_2_2_stats    | 2024-11-03 12:59:54 |              720 |        0.542773 |           0.542773 | 2024-11-03 19:31:07 |
| ConvLSTM_IQ_SNR_k7_k3_stats                         | 2024-10-28 22:07:25 |             1860 |        0.538    |           0.536886 | 2024-10-29 08:38:05 |
| rnn_lstm_w_SNR2_stats                               | 2024-11-07 13:47:59 |             1080 |        0.536773 |           0.494818 | 2024-11-08 09:04:42 |
| ConvLSTM_IQ_SNR_stats                               | 2024-10-28 20:12:50 |              340 |        0.534955 |           0.534932 | 2024-10-28 21:32:15 |
| ConvLSTM_IQ_SNR_k7_stats                            | 2024-10-28 21:37:37 |              380 |        0.529955 |           0.529955 | 2024-10-29 10:50:41 |
| ConvLSTM_IQ_SNR_k7_BW_stats                         | 2024-10-29 08:39:46 |             2600 |        0.528227 |           0.527136 | 2024-10-29 21:58:18 |
| rnn_lstm_multifeature_generic_5_2_1_stats           | 2024-10-27 21:52:19 |              280 |        0.527705 |           0.511818 | 2024-10-28 17:45:45 |
| rnn_lstm_multifeature_generic_w_fft_stats           | 2024-10-25 21:57:40 |              880 |        0.517455 |           0.515477 | 2024-10-26 11:55:08 |
| EnhancedModulationClassifier_KitchenSink_stats      | 2024-11-05 10:12:55 |              160 |        0.469886 |           0.466159 | 2024-11-05 13:03:33 |
| ConvLSTM_FFT_Power_SNR_stats                        | 2024-10-29 16:07:00 |             1680 |        0.433295 |           0.432295 | 2024-10-30 15:54:34 |
| EnhancedModulationClassifier_AMvsPSK_stats          | 2024-11-02 13:31:44 |             6980 |        0.407659 |           0.406386 | 2024-11-03 10:30:06 |

# Next Steps...

I am working on a classifier for WBFM versus Others and may implement an ensemble model to achieve superior accuracy


# Troubleshooting if running on NVIDIA Jetson Xavier

If you see this error message:
ImportError: /home/dev/python/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block

the fix for me is to run export LD_PRELOAD=/home/dev/python/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

this way the library is loaded before python loads scikit-learn

# References

Bai, H., Huang, M., & Yang, J. (2023, October 1). An efficient Automatic Modulation Classification method based on the Convolution Adaptive Noise Reduction network. ICT Express, pp. 834-840. Retrieved from https://www.sciencedirect.com/science/article/pii/S2405959522001515

​​Bhuiya, S. (2020, October 31). Disadvantages of CNN models. Retrieved from Medium: https://sandeep-bhuiya01.medium.com/disadvantages-of-cnn-models-95395fe9ae40 

​Choudhary, J. (2023, September 20). Mastering XGBoost: A Technical Guide for Machine Learning Practitioners. Retrieved from Medium: https://medium.com/@jyotsna.a.choudhary/mastering-xgboost-a-technical-guide-for-intermediate-machine-learning-practitioners-f7ad167c6865 

​DeepSig. (2024, October 1). Datasets. Retrieved from DeepSig AI: https://www.deepsig.ai/datasets/ 

​Duda, R. O., Hart, P. E., & Stork, D. G. (2001). Pattern Classification (2nd ed.). New York: John Wiley & Sons, Inc. 

​Flowers, B., & Headly, W. C. (2024, September 24). Radio Frequency Machine Learning (RFML) in PyTorch. Retrieved from Github: https://github.com/brysef/rfml 

​GeeksForGeeks. (2024, October 10). Support Vector Machine (SVM) Algorithm. Retrieved from GeeksForGeeks: https://www.geeksforgeeks.org/support-vector-machine-algorithm/ 

​Gish, H. (2006, August 06). A probabilistic approach to the understanding and training of neural network classifiers. International Conference on Acoustics, Speech, and Signal Processing. Albuquerque: IEEE Xplore. Retrieved from IEEE Explore: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=115636 

​Google Developers. (2022, September 28). Gradient Boosted Decision Trees. Retrieved from Google Developers.: https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt 

​Hastie, T., Tibshirani, R., & Friedman, J. (2001). The Elements of Statistical Learning. Springer. 

​Haykin, S. (1999). Neural Networks: A Comprehensive Foundation. Prentice-Hall, Inc. 

​Hutomo, I. S. (2021, October 8). https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification. Retrieved from Github: https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification 

​IBM. (2024, October 9). What is a decision tree? Retrieved from IBM: https://www.ibm.com/topics/decision-trees 

​IBM. (2024, October 9). What is random forest? Retrieved from IBM: https://www.ibm.com/topics/random-forest 

​Keylabs AI. (2024, September 13). K-Nearest Neighbors (KNN): Real-World Applications. Retrieved from Keylabs AI: https://keylabs.ai/blog/k-nearest-neighbors-knn-real-world-applications/ 

​Kızılırmak, S. (2023, February 11). Rectified Linear Unit (ReLU) Function: Understanding the Basics. Retrieved from Medium: https://medium.com/@serkankizilirmak/rectified-linear-unit-relu-function-in-machine-learning-understanding-the-basics-3770bb31c2a8 

​O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). Over-the-Air Deep Learning Based Radio Signal Classification. IEEE Journal of Selected Topics in Signal Processing, 168-179. 

​O'Shea, T., & West, N. (2016, September 6). Radio Machine Learning Dataset Generation with GNU Radio. Retrieved from Gnuradio: https://pubs.gnuradio.org/index.php/grcon/article/view/11/10 

​Qiu, Y., Zhang, J., Chen, Y., & Zhang, J. (2023, April 20). Radar2: Passive Spy Radar Detection and Localization Using COTS mmWave Radar. Retrieved from IEEE Explore: https://ieeexplore.ieee.org/document/10105863 

​Roy, D. (2020). MACHINE LEARNING BASED RF TRANSMITTER CHARATERIZATION IN THE. Retrieved from Northeastern University College of Engineering: https://www1.coe.neu.edu/~droy/Doctoral_Dissertation_Debashri_Roy.pdf 

​Virginia Tech. (2024, October 9). ECE5424 LN12.pdf. Blackburg, VA, United Stated of America. 

​Viso.AI. (2024, October 9). Ensemble Learning: A Combined Prediction Model (2024 Guide). Retrieved from Viso.AI: https://viso.ai/deep-learning/ensemble-learning/ 

​​​ 
