This is a repository to hold the code for our research paper: Wireless Signal Classification via Machine Learning.

Download the RADIOML 2016.10A dataset from this link https://www.deepsig.ai/datasets/

Extract the dataset and it will be loaded on startup

After the application starts, it will create a model if one isn't present, train it, validate it, then save itself for next run


To use with a GPU:
https://www.tensorflow.org/install/source#gpu

I'm using these versions:

tensorflow-2.12.0
Python 3.10	
cuDNN 8.6
CUDA 11.8


Look under the RNN/stats folder for the current best accuracy statistics. Confusion matrixes coming soon