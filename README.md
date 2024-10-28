# Purpose

This is a repository to hold the code for our research paper: Wireless Signal Classification via Machine Learning.

# Datasets

Download the RADIOML 2016.10A dataset from this link https://www.deepsig.ai/datasets/

Extract the dataset and it will be loaded on startup

# What does the SW do?
After the application starts, it will create a model if one isn't present, train it, validate it, then save itself for next run

Look under the RNN folder for two models I'm experimenting with


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


# References

https://github.com/radioML

https://github.com/BolognaBiocomp/deepsig

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