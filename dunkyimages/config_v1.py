# import the necessary packages
import os

# initialize the list of class label names
CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]

# initialize the path to the input directory containing our dataset
# of images
DATASET_PATH = "dataset_c"

# initialize the number of image dimensions
IMAGE_DIMS = (32, 32, 3)

# size of the training and validation set
# train-test-split.
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 128 
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plotv1.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot_LRF.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])



        
