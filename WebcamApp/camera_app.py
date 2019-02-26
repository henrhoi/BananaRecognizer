from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import time
import os
from banana_predicter import build_label
import cv2
import imutils
from imutils.video import WebcamVideoStream

# Application setup
MODEL_NAME = '../BananaNetwork/le_banana_net_26_02_2019.h5'
INPUT_SHAPE = (64, 64)
PROBABILISTIC = True

# Initializing global variables
CON_BANANA_FRAMES = 0
BANANA = False
CON_TRESH = 200
YELLOW = (0, 255, 255)

# Starting application
print("[INFO] Press [Q] or [ESC] to quit camera application")
print("[INFO] Starting stream from web camera...")
video_stream = WebcamVideoStream(src=0).start()
model = load_model(os.path.abspath(MODEL_NAME))
time.sleep(1.0)

while True:
	# Grab frame from the threaded video stream and resize it
	frame = video_stream.read()
	frame = imutils.resize(frame, width=500)

	# Prepare the image to be classified by Banana Network
	image = cv2.resize(frame, INPUT_SHAPE)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# Classify the input image and initialize the label and probability of the prediction, if model is probabilistic
	result = model.predict(image)
	label, is_banana = build_label(result, banana_label="Banana", probabilistic=PROBABILISTIC)

	# Counting consecutive frames with bananas
	CON_BANANA_FRAMES += 1 if is_banana else 0

	if is_banana and not BANANA and CON_BANANA_FRAMES >= CON_TRESH:
		# Banana has been found
		BANANA = True

	if not is_banana:
		# Banana has been lost
		TOTAL_CONSEC = 0
		BANANA = False

	# Add prediction-label to frame
	frame = cv2.putText(frame, label, (140, 260), cv2.FONT_HERSHEY_DUPLEX, 0.7, YELLOW, 1)

	# Draw yellow border if banana is found
	frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=YELLOW) if BANANA else frame

	cv2.imshow("Banana/Not Banana", frame)
	key = cv2.waitKey(1) & 0xFF

	# Stop stream prediction if 'q' or 'ESC' is pressed
	if key == ord("q") or key == 27:
		break

# Stop stream and destroy cv2's windows
cv2.destroyAllWindows()
video_stream.stop()
