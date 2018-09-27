# Chase Brown
# SID 106015389
# DeepLearning PA 2: ANN/CNN using Keras

import numpy as np
import data_process
import nn_models
from miscfunctions import MiscFunctions as mf
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from keras.datasets import mnist
# Fix random seed for reducibility

if __name__ == '__main__':
    # only way I was able to fix the pathing issue with Graphviz and windows 10
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    # # Run ANN model
    nn_models.NnModels.baseline_ann()
    # # Run baseline CNN model
    nn_models.NnModels.baseline_cnn()
    # Run bigger CNN model
    nn_models.NnModels.bigger_cnn()

    print('Test')
