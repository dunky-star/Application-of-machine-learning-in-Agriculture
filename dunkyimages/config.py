#import the necessary packages
import os

# initialize the list of class label names
CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]

# initialize the path to the input directory containing our dataset
# of images
DATASET_PATH = "dataset"

# initialize the number of image dimensions
IMAGE_DIMS = (96, 96, 3)

# size of the training and validation set
# train-test-split.
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-6
MAX_LR = 1e-3
BATCH_SIZE = 64 
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 164
	
# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "cassava_mosaic4_focalloss.model"])

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot_focalloss.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot_LRF.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "lr_plot.png"])



        
