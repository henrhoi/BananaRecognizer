# Inspired from LeNet-5 LeCun 1998

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam


def build_network(save_path, input_shape, optimizer, classes, save=True):
	model = Sequential()

	# first set of CONV => RELU => POOL layers
	model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL layers
	model.add(Conv2D(50, (5, 5), padding="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

	model.save(save_path) if save else None

	return model


height, width, depth = 28, 28, 3
input_shape = (height, width, depth)

CLASSES, EPOCHS, INIT_LR = 2, 25, 1e-3

optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

build_network('BananaNetwork/le_banana_net.h5', input_shape, optimizer, CLASSES)
