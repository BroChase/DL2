# Chase Brown
# SID 106015389
# DeepLearning PA 2: CNN using Keras

# cnn_models: contains cnn models
from keras import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras import backend as K
K.set_image_dim_ordering('tf')


class CnnModels:
    @staticmethod
    def baseline_ann(num_of_pixels, num_of_classes):
        model = Sequential()
        model.add(Dense(num_of_pixels, input_dim=num_of_pixels,
                        kernel_initializer='normal', activation='tanh', name='Hidden_layer_1'))
        model.add(Dense(512, kernel_initializer='normal', activation='sigmoid', name='Hidden_layer_2'))
        model.add(Dense(100, kernel_initializer='normal', activation='relu', name='Hidden_layer_3'))
        model.add(Dense(num_of_classes, kernel_initializer='normal', activation='softmax', name='Output_layer'))
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def baseline_cnn(num_of_classes):
        model = Sequential()
        # First hidden layer
        model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(1, 28, 28),activation='relu'))
        # Pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening layer
        model.add(Flatten())
        # Fully connected layer
        model.add(Dense(units=128, activation='relu'))
        # Output layer
        model.add(Dense(units=num_of_classes, activation='softmax'))
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # return the model
        return model
