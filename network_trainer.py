from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NETWORK = 'BananaNetwork/banana_net.h5'

# Defining dataset paths
SYNSET_FRUIT365_TRAINING = 'datasets/synset_fruit365/training'
SYNSET_FRUIT365_VALIDATION = 'datasets/synset_fruit365/validation'

GOOGLE_UKBENCH_TRAINING = 'datasets/google_images/training'
GOOGLE_UKBENCH_VALIDATION = 'datasets/google_images/validation'

SYNSET_FRUIT365_GOOGLE_UKBENCH_TRAINING = 'datasets/google_synset_fruit365/training'
SYNSET_FRUIT365_GOOGLE_UKBENCH_VALIDATION = 'datasets/google_synset_fruit365/validation'


# Defining datasets
dataset_training_path = SYNSET_FRUIT365_GOOGLE_UKBENCH_TRAINING
dataset_validation_path = SYNSET_FRUIT365_GOOGLE_UKBENCH_VALIDATION

# Because of uneven datasets
epochs_steps = min(len(os.listdir(dataset_training_path + "/banana")),
				   len(os.listdir(dataset_training_path + "/other")))
validation_steps = min(len(os.listdir(dataset_validation_path + "/banana")),
					   len(os.listdir(dataset_validation_path + "/other")))

# Retrieving model
model = load_model(DEFAULT_NETWORK)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
								   rotation_range=30, width_shift_range=0.1,
								   height_shift_range=0.1, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(SYNSET_FRUIT365_TRAINING,
												 target_size=(64, 64),
												 batch_size=32,
												 class_mode='binary',
												 shuffle=True)

test_set = test_datagen.flow_from_directory(SYNSET_FRUIT365_VALIDATION,
											target_size=(64, 64),
											batch_size=32,
											class_mode='binary')

# Defining epochs, trains model, and saves
EPOCHS = 5

earlystop = EarlyStopping(monitor='acc', baseline=1.0, patience=0)

train_history = model.fit_generator(training_set,
									steps_per_epoch=epochs_steps,
									epochs=EPOCHS,
									validation_data=test_set,
									validation_steps=validation_steps)

model.save(DEFAULT_NETWORK)

# Plotting training epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), train_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), train_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), train_history.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), train_history.history["val_acc"], label="val_acc")
plt.title("Loss/Accuracy on Banana/Not Banana")
plt.xlabel("epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('epoch_fig')
