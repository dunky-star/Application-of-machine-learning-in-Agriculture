################################################################
#                                                              # 
# Written  by Opiyo Geoffrey Duncan: Deep Learning             #
################################################################
# USAGE
# python train_LRF.py --lr-find 1

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from dunkyimages.callbacks.epochcheckpoint import EpochCheckpoint
from dunkyimages.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from dunkyimages.learningratefinder import LearningRateFinder
from keras.utils import to_categorical
from dunkyimages.clr_callback import CyclicLR
from sklearn.utils import class_weight
from dunkyimages import config
from dunkyimages import focalloss
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dunkyimages.cassavavggnet import CassavaVGGNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from keras.models import load_model
from imblearn.over_sampling import SMOTE
import numpy as np
import keras.backend as K
from imutils import paths
import argparse
import random
import pickle
import cv2
import os
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

def load_dataset(datasetPath):
	# grab the paths to all images in our dataset directory, then
	# initialize our lists of images
	imagePaths = list(paths.list_images(datasetPath))
	data = []
 
	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# load the image and resize it to be a fixed 128x128 pixels,
		# ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
		image = cv2.resize(image, (96, 96))
 
		# add the image to the data lists
		data.append(image)
		if i > 0 and i % 1000 == 0:
			print("[INFO] processed {}/{}".format(i, len(imagePaths)))
 
	# return the data list as a NumPy array
	return np.array(data, dtype="float32")
# load the fire and non-fire images
print("[INFO] loading data...")
infectedData = load_dataset(config.DATASET_PATH_INFECTED)
healthyData = load_dataset(config.DATASET_PATH_HEALTHY)
 
# construct the class labels for the data
infectedLabels = np.ones((infectedData.shape[0],))
healthyLabels = np.zeros((healthyData.shape[0],))
 
# stack the fire data with the non-fire data, then scale the data
# to the range [0, 1]
data = np.vstack([infectedData, healthyData])
labels = np.hstack([infectedLabels, healthyLabels])
data /= 255
labels = to_categorical(labels, num_classes=2)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# account for skew/imbalanced data in the image
#classTotals = labels.sum(axis=0)
#ClassWeight = classTotals.max() / classTotals
# Reshaping ndarray to work with SMOTE algorithm
#nx = np.array([[100], [100],[3]])
#nsamples, nx, ny, nz = data.shape
#train_data = data.reshape(nsamples,nx*ny*nz)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=config.TEST_SPLIT, random_state=42)


print('\n\n')
print('Training shape before:')
print (trainX.shape, trainY.shape)
print('\n\n')

sm = SMOTE('minority')

trainX_res, trainY_res = sm.fit_resample(trainX, trainY)

print('\n\n')
print('Training shape after SMOTE:')
print (trainX_res.shape, trainY_res.shape)
print('\n\n')


# construct the image generator for data augmentation
aug = ImageDataGenerator(
			rotation_range=30,
        		zoom_range=0.15,
        		width_shift_range=0.2,
        		height_shift_range=0.2,
        		shear_range=0.15,
        		horizontal_flip=True,
        		vertical_flip=True,
        		fill_mode="nearest"
			)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=config.MIN_LR)
#opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = CassavaVGGNet.build(width=config.IMAGE_DIMS[1], height=config.IMAGE_DIMS[0],
		depth=config.IMAGE_DIMS[2], classes=2)
#model.compile(loss="categorical_crossentropy", optimizer=opt,
#	metrics=["accuracy"])

model.compile(optimizer=opt, loss = binary_crossentropy, metrics = ['accuracy'])

# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs
if args["lr_find"] > 0:
	# initialize the learning rate finder and then train with learning
	# rates ranging from 1e-10 to 1e+1
	print("[INFO] finding learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(
		aug.flow(trainX_res, trainY_res, batch_size=config.BATCH_SIZE),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil((trainX_res.shape[0] / float(config.BATCH_SIZE))),
		epochs=config.NUM_EPOCHS,
		batchSize=config.BATCH_SIZE,
		class_weight = ClassWeight
		)
 
	# plot the loss for the various learning rates and save the
	# resulting plot to disk
	lrf.plot_loss()
	plt.savefig(config.LRFIND_PLOT_PATH)
 
	# gracefully exit the script so we can adjust our learning rates
	# in the config and then train the network for our full set of
	# epochs
	print("[INFO] learning rate finder complete")
	print("[INFO] examine plot and adjust learning rates before training")
	sys.exit(0)


# otherwise, we have already defined a learning rate space to train
# over, so compute the step size and initialize the cyclic learning
# rate method
stepSize = config.STEP_SIZE * (trainX_res.shape[0] // config.BATCH_SIZE)
clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size=stepSize)

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX_res, trainY_res, batch_size=config.BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=trainX_res.shape[0] // config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	callbacks=[clr],
	shuffle=True,
	class_weight = ClassWeight,
	verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=config.BATCH_SIZE)

print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=config.CLASSES))



# serialize the model to disk
print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

# construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(config.TRAINING_PLOT_PATH)
# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)


