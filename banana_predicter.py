import numpy as np
import os
from keras.models import load_model
from keras_preprocessing import image

DEFAULT_NETWORK = os.getcwd() + "/BananaNetwork/le_banana_net_26_02_2019.h5"
TARGET_SIZE = (64, 64)


def build_label(result, banana_label='🍌', probabilistic=True):
	"""
	Builds prediction-label from predict-result
	:param banana_label: Label for banana in result
	:param probabilistic: Whether or not the output-layer is binary or with probabilistic value
	:param result: result from .predict()
	:return: label {String}, True/False {Banana/Not Banana}
	"""
	if not probabilistic:
		label, is_banana = (banana_label, True) if result[0][0] == 0 else ('Not Banana', False)
	else:
		(banana, not_banana) = result[0]
		label, is_banana = (banana_label, True) if not_banana < banana else ("Not Banana", False)
		proba = not_banana if not_banana > banana else banana
		label = "{}: {:.2f}%".format(label, proba * 100)

	return label, is_banana


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
		try:
			test_image = image.load_img("{}/{}".format(folder_path, test_images[i]), target_size=TARGET_SIZE)
			test_image = image.img_to_array(test_image)
			test_image = np.expand_dims(test_image, axis=0)
			result = model.predict(test_image)

			label, is_banana = build_label(result)
			bananas += 1 if is_banana else 0
			print("Predicting [{}] as {}".format(test_images[i], label))

		except OSError or TypeError:
			print("[ERROR] Failed to predict {}".format(test_images[i]))
			continue

	print('{} 🍌\'s in the {} pictures'.format(bananas, len(test_images)))


predict_images("predict_tests", DEFAULT_NETWORK)


def predict_image(image_path, model_name=DEFAULT_NETWORK):
	model = load_model(model_name)

	print("Predicting [" + image_path.split("/")[-1] + "] as ", end="")
	test_image = image.load_img(image_path, target_size=TARGET_SIZE)
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict(test_image)

	label = build_label(result)
	print(label)

	return result
