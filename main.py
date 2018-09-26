# Chase Brown
# SID 106015389
# DeepLearning PA 2: CNN using Keras

import numpy as np
import data_process
import nn_models
from miscfunctions import MiscFunctions as mf
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
# Fix random seed for reducibility

if __name__ == '__main__':
    # only way I was able to fix the pathing issue with Graphviz and windows 10
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    seed = 106015389
    np.random.seed(seed)

    # Load Data
    ld = data_process.LoadDataModule()
    # Now let's load the dataset
    x_train, y_train = ld.load('train')
    x_test, y_test = ld.load('test')

    # Plot the first image
    # plt.imshow(np.reshape(x_train[0, :], (28, 28)))

    # Min max the data from 0-255 'gray scale' to 0-1
    x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
    # Class names 0-9
    class_names = np.unique(y_train)
    # One-hot encode y_train and y_test
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Number of classes
    num_of_classes = y_test.shape[1]

    # build the model
    model = nn_models.CnnModels.baseline_ann(x_train.shape[1], num_of_classes)

    # Print the model
    print(model.summary())
    plot_model(model, to_file='ann_model_plot.png', show_shapes=True, show_layer_names=True)
    im = cv2.imread('ann_model_plot.png')
    height, width, channels = im.shape
    print("Height = %d, Width = %d, Channels = %d" % (height, width, channels))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

    t = time.localtime(time.time())
    timeStamp = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) \
                + '--' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec)
    print(timeStamp)

    tBoard = TensorBoard(log_dir='logs/{}'.format(timeStamp))
    # Fit the model, and record history of training results
    # define the params
    num_epochs = 10
    batch_size = 200
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs,
                        batch_size=batch_size, verbose=2, callbacks=[tBoard])

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Baseline error: %.2f' % (1 - scores[1]))
    print("Accuracy: %.2f" % scores[1])

    # Print/plot the training history
    mf.plot_history(history)

    y_pred = model.predict_classes(x_test)
    y_pred_probabilities = model.predict(x_test)

    y_test_original = np.argmax(y_test, axis=1)

    print("Classification report \n=======================")
    print(classification_report(y_true=y_test_original, y_pred=y_pred))
    print("Confusion matrix \n=======================")
    print(confusion_matrix(y_true=y_test_original, y_pred=y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=y_test_original, y_pred=y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    mf.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    mf.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

    plt.show()
    print('Test')
