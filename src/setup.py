import sys
import os

# SRC_CONSTANTS
BATCH_SIZE = 10
NUM_EPOCHS = 50
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 650
LEARNING_RATE=0.01

# PATH CONSTANTS
SRC_PATH = os.path.dirname(__file__)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, 'dataset', 'Covid19-dataset')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
