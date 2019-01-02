# Importing Keras libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Constructing Sequential Convolutional Neural Network
def build_network(save_path, save=True):
	classifier = Sequential()

	# Explain further steps
	classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
	classifier.add(MaxPooling2D(pool_size=(2, 2)))
	classifier.add(Flatten())
	classifier.add(Dense(units=128, activation='relu'))

	# Single output node, because it is only a binary classification problem (banana or not banana)
	classifier.add(Dense(units=1, activation='sigmoid'))

	# Using binary cross entropy as loss-function because of binary classification
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# Saving model to file
	classifier.save(save_path) if save else None
	return classifier


build_network('BananaNetwork/banana_net.h5')
