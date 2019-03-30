import imghdr
import tkinter.filedialog as tkFileDialog
import tkinter as tk
import cv2
import pygubu
from PIL import ImageTk, Image
import os
from banana_predicter import predict_image


class GUIPredicter:
	"""
	Class that uses tKinter and PyGubu-Builder for UI to predicting images
	"""

	def __init__(self, root):
		self.builder = builder = pygubu.Builder()
		builder.add_from_file('gui.ui')
		self.mainframe = builder.get_object('MainFrame', root)

		builder.connect_callbacks(self)
		callbacks = {'openFile': self.openFile}
		builder.connect_callbacks(callbacks)

	def openFile(self):
		"""
		Predicts chosen file
		"""
		filename = tkFileDialog.askopenfilename()
		image_type = imghdr.what(filename)

		if image_type is None:
			print("[ERROR] Invalid image type")
			return

		if filename:
			result = predict_image(filename, model_name="../BananaNetwork/banana_net_17_12_18.h5", probabilistic=False)
			prediction = 'Banana' if result[0][0] == 0 else 'Not Banana'

			# Showing prediction
			label = self.builder.get_object('Prediction')
			label.configure(text='Prediction: ' + prediction)

			image_label = self.builder.get_object('Image')
			img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

			# Resizing and showing image with OpenCV and PIL
			resize_scale = img.shape[1] / 300
			width = int(img.shape[1] / resize_scale)
			height = int(img.shape[0] / resize_scale)
			dim = (width, height)
			img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

			img = ImageTk.PhotoImage(image=Image.fromarray(img))
			image_label.configure(image=img, width=width, height=height)
			image_label.image = img


if __name__ == '__main__':
	root = tk.Tk()
	root.title("Banana or Not Banana")
	GUIPredicter(root)
	root.mainloop()
