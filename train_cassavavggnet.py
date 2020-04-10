################################################################
# Written  by Opiyo Geoffrey Duncan: Deep Learning             #
################################################################

# Setting matplotlib backend so that figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# Import the necessary packages
from dunkyimages.cassavavggnet import CassavaVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from dunkyimages.learningratefinder import LearningRateFinder
from dunkyimages.clr_callback import CyclicLR
from sklearn.utils import class_weight
from dunkyimages import config
from dunkyimages import focalloss
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dunkyimages.smallervggnet import SmallerVGGNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 124
INIT_LR = 1e-3
BS = 64

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
#random.seed(42)
#random.shuffle(imagePaths)

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

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeights = classTotals.max() / classTotals

# construct the training and testing split
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=config.TEST_SPLIT, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")


# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = CassavaVGGNet.build(width=config.IMAGE_DIMS[1], height=config.IMAGE_DIMS[0], depth=config.IMAGE_DIMS[2]
                         ,classes=len(lb.classes_))
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeights,
	#shuffle = True,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=config.CLASSES))
 
# serialize the model to disk
print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

# plot the training loss and accuracy
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(config.TRAINING_PLOT_PATH)

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Learning Rate (LR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)



