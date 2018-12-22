import numpy as np
import os
from keras.models import load_model
from keras_preprocessing import image


DEFAULT_NETWORK = os.getcwd() + "/banana_net.h5"


def predict_images(folder_path, model_name):
	"""
	Method for predicting all images in a given folder path
	:param folder_path: Relative or absolute path to images to be predicted
	:param model_name: Name of model for prediction
	"""
	model = load_model(model_name)
	test_images = os.listdir(folder_path)

	print("Images to predict: " + ", ".join(test_images))

	bananas = 0
	for i in range(len(test_images)):
		print("Predicting [" + test_images[i] + "] as ", end="")
		test_image = image.load_img("{}/{}".format(folder_path, test_images[i]), target_size=(64, 64))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis=0)
		result = model.predict(test_image)

		prediction = '🍌' if result[0][0] == 0 else 'Not Banana'
		bananas += 1 if result[0][0] == 0 else 0
		print(prediction)

	print('{} 🍌\'s in the {} pictures'.format(bananas, len(test_images)))


#predict_images("predict_tests", DEFAULT_NETWORK)


def predict_image(image_path, model_name=DEFAULT_NETWORK):
	model = load_model(model_name)

	print("Predicting [" + image_path.split("/")[-1] + "] as ", end="")
	test_image = image.load_img(image_path, target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict(test_image)[0][0]

	print('🍌' if result == 0 else 'Not Banana')
	return result


