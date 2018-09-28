# Chase Brown
# SID 106015389
# DeepLearning PA 2: ANN/CNN using Keras

import nn_models
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
print(tf.__version__)

if __name__ == '__main__':
    # only way I was able to fix the pathing issue with Graphviz and windows 10
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    # Run ANN model
    nn_models.NnModels.baseline_ann()
    # Run baseline CNN model
    nn_models.NnModels.baseline_cnn()
    # Run bigger CNN model
    nn_models.NnModels.bigger_cnn()

