#################################################################
#                                                               # 
# Written  by Geoffrey Duncan Opiyo : Deep Learning Practitioner#
#################################################################
# USAGE
# python train_LRF.py --model cassava_detect.model 

# setting the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# importing the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from dunkyimages.callbacks.epochcheckpoint import EpochCheckpoint
from dunkyimages.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.optimizers import SGD
from dunkyimages.learningratefinder import LearningRateFinder
from dunkyimages.clr_callback import CyclicLR
from sklearn.utils import class_weight
from dunkyimages import config
from dunkyimages import focalloss
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dunkyimages.cassavavggnet import CassavaVGGNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from imutils import paths
import numpy as np
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

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(config.DATASET_PATH)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_DIMS[1], config.IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
        # show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#classTotals = labels.sum(axis=0)
#classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=config.TEST_SPLIT, random_state=42)

# account for skew/imbalanced data in the image
'''
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(trainY),
                                                 trainY)
'''

# construct the image generator for data augmentation
aug = ImageDataGenerator(
			rotation_range=25,
		        width_shift_range=0.1,
			height_shift_range=0.1,
		        shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode="nearest"
			)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=config.MIN_LR)
#opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = SmallerVGGNet.build(width=config.IMAGE_DIMS[1], height=config.IMAGE_DIMS[0],
		depth=config.IMAGE_DIMS[2], classes=5)
history = model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#history = model.compile(optimizer=opt, loss = focalloss.focal_loss(alpha=1), metrics = ['accuracy'])

# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs
if args["lr_find"] > 0:
	# initialize the learning rate finder and then train with learning
	# rates ranging from 1e-10 to 1e+1
	print("[INFO] finding learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(
		aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
		epochs=124,
		batchSize=config.BATCH_SIZE,
		#classWeight=classWeight
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
stepSize = config.STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size=stepSize)

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	callbacks=[clr],
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
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
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

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

