from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np


# Defining dataset paths
synset_fruit365_training = 'synset_fruit365/training'
synset_fruit365_validation = 'synset_fruit365/validation'

google_ukbench_training = 'google_images/training'
google_ukbench_validation = 'google_images/validation'

synset_fruit365_google_ukbench_training = 'google_synset_fruit365/training'
synset_fruit365_google_ukbench_validation = 'google_synset_fruit365/validation'

# Defining datasets
dataset_training_path = synset_fruit365_google_ukbench_training
dataset_validation_path = synset_fruit365_google_ukbench_validation

# Because of uneven datasets
epochs_steps = min(len(os.listdir(dataset_training_path + "/banana")),
				   len(os.listdir(dataset_training_path + "/other")))
validation_steps = min(len(os.listdir(dataset_validation_path + "/banana")),
					   len(os.listdir(dataset_validation_path + "/other")))

# Retrieving model
model = load_model('BananaNetwork/banana_net_19_12_18.h5')

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
								   rotation_range=30, width_shift_range=0.1,
								   height_shift_range=0.1, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(synset_fruit365_training,
												 target_size=(64, 64),
												 batch_size=32,
												 class_mode='binary',
												 shuffle=True)
test_set = test_datagen.flow_from_directory(synset_fruit365_validation,
											target_size=(64, 64),
											batch_size=32,
											class_mode='binary',
											shuffle=True)

# Defining epochs, trains model, and saves
EPOCHS = 5

earlystop = EarlyStopping(monitor='acc', baseline=1.0, patience=0)

train_history = model.fit_generator(training_set,
									steps_per_epoch=epochs_steps,
									epochs=EPOCHS,
									validation_data=test_set,
									validation_steps=validation_steps)

model.save('BananaNetwork/banana_net_19_12_18.h5')

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
