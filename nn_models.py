# Chase Brown
# SID 106015389
# DeepLearning PA 2: ANN/CNN using Keras

# cnn_models: contains cnn models
import numpy as np
import data_process
from keras import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import time
from miscfunctions import MiscFunctions as mf
from keras import backend as K
K.set_image_dim_ordering('tf')


class NnModels:
    @staticmethod
    def baseline_ann():
        # Random seed with SID
        seed = 106015389
        np.random.seed(seed)

        # Load Data
        ld = data_process.LoadDataModule()

        # Load Data into training/testing sets
        x_train, y_train = ld.load('train')
        x_test, y_test = ld.load('test')

        # Min max the data from 0-255 'gray scale' to 0-1
        x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
        x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

        # Class names 0-9
        class_names = np.unique(y_train)

        # One-hot encode y_train and y_test
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # Number of classes
        num_of_classes = y_test.shape[1]
        # Number of pixels
        num_of_pixels = x_train.shape[1]


        model = Sequential()
        model.add(Dense(num_of_pixels, input_dim=num_of_pixels,
                        kernel_initializer='normal', activation='tanh', name='Hidden_layer_1'))
        model.add(Dense(512, kernel_initializer='normal', activation='sigmoid', name='Hidden_layer_2'))
        model.add(Dense(100, kernel_initializer='normal', activation='relu', name='Hidden_layer_3'))
        model.add(Dense(num_of_classes, kernel_initializer='normal', activation='softmax', name='Output_layer'))
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Print the model
        mf.model_summary(model, 'ann_model_plot.png')

        t = time.localtime(time.time())
        timeStamp = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) \
                    + '--' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec)
        # Create a TensorBoard
        tBoard = TensorBoard(log_dir='logs/{}'.format(timeStamp))

        # Define params for model fit
        num_epochs = 50
        batch_size = 200
        # Fit the model and record the history of the training results
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs,
                            batch_size=batch_size, verbose=2, callbacks=[tBoard])

        # Evaluate the model
        mf.final_eval(model, x_test, y_test, history, class_names, 'ann_model')

    @staticmethod
    def baseline_cnn():
        # Random seed with SID
        seed = 106015389
        np.random.seed(seed)

        # Load Data
        ld = data_process.LoadDataModule()
        # Load Data into training/testing sets
        x_train, y_train = ld.load('train')
        x_test, y_test = ld.load('test')

        x_train = mf.reshape_x(x_train)
        x_test = mf.reshape_x(x_test)

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
        x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

        class_names = np.unique(y_train)

        # one hot encoding of the targets
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        num_of_classes = y_test.shape[1]

        model = Sequential()
        # First hidden layer: 40 filters, kernel 5x5, activation rectified linear, stides 1, padding none
        model.add(Conv2D(filters=40, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu', strides=1,
                         padding='valid'))
        # Pooling layer: Max pooling layer with 2x2 size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening layer for input to connected layer
        model.add(Flatten())
        # Fully connected layer: 200 neurons, activation rectified linear
        model.add(Dense(units=200, activation='relu'))
        # Output layer: number of classed 0-9, activation softmax
        model.add(Dense(units=num_of_classes, activation='softmax'))
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print the model
        mf.model_summary(model, 'cnn_model_plot.png')

        t = time.localtime(time.time())
        timeStamp = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) \
                    + '--' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec)
        # Create a TensorBoard
        tBoard = TensorBoard(log_dir='logs/{}'.format(timeStamp))

        # Define params for model fit
        num_epochs = 50
        batch_size = 200

        # Fit the model and record the history of the training results
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs,
                                batch_size=batch_size, verbose=2, callbacks=[tBoard])

        # Evaluate the model
        mf.final_eval(model, x_test, y_test, history, class_names, 'cnn_model')

    @staticmethod
    def bigger_cnn():
        # Random seed with SID
        seed = 106015389
        np.random.seed(seed)

        # Load Data
        ld = data_process.LoadDataModule()
        # Load Data into training/testing sets
        x_train, y_train = ld.load('train')
        x_test, y_test = ld.load('test')

        x_train = mf.reshape_x(x_train)
        x_test = mf.reshape_x(x_test)

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
        x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

        class_names = np.unique(y_train)

        # one hot encoding of the targets
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        num_of_classes = y_test.shape[1]

        model = Sequential()
        # First hidden layer: 40 filters, kernel 3x3, activation rectified linear, stides 1, padding none
        model.add(Conv2D(filters=48, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', strides=1,
                         padding='valid'))
        # Second Hidden Layer: Pooling layer: Max pooling layer with 2x2 size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Third hidden layer: 96 filters, kernel 3x3, activation rectified linear, stides 1, padding none
        model.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu', strides=1,
                         padding='valid'))
        # Fourth hidden layer: Pooling layer: Max pooling layer with 2x2 size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening layer for input to connected layer
        model.add(Flatten())
        # Fully connected layer: 100 neurons, activation rectified linear
        model.add(Dense(units=100, activation='relu'))
        # Output layer: number of classed 0-9, activation softmax
        model.add(Dense(units=num_of_classes, activation='softmax'))
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print the model
        mf.model_summary(model, 'cnn_model_plot.png')

        t = time.localtime(time.time())
        timeStamp = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) \
                    + '--' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec)
        # Create a TensorBoard
        tBoard = TensorBoard(log_dir='logs/{}'.format(timeStamp))

        # Define params for model fit
        num_epochs = 50
        batch_size = 200

        # Fit the model and record the history of the training results
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs,
                                batch_size=batch_size, verbose=2, callbacks=[tBoard])

        # Evaluate the model
        mf.final_eval(model, x_test, y_test, history, class_names, 'big_cnn_model')



