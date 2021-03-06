# Inspired from LeNet-5 LeCun 1998

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
import keras
import tensorflow as tf


config = tf.ConfigProto()

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
keras.backend.set_session(tf.Session(config=config))

def build_network(save_path, input_shape, optimizer, classes, save=True):
    model = Sequential()

    # First set of CONV > RELU > POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second set of CONV > RELU > POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # First (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # Softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.save(save_path) if save else None

    return model


def main():
    height, width, depth = 64, 64, 3
    input_shape = (height, width, depth)

    CLASSES, EPOCHS, INIT_LR = 2, 25, 1e-3

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    build_network('BananaNetwork/le_banana_net_26_02_2019.h5', input_shape, optimizer, CLASSES)


if __name__ == '__main__':
    main()
