import numpy as np
import os
from keras.models import load_model
from keras_preprocessing import image


def predict_images(folder_path, model_name):
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


predict_images("predict_tests", 'BananaNetwork/banana_net_19_12_18.h5')
