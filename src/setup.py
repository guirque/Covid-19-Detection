import sys
import os

# SRC_CONSTANTS
BATCH_SIZE = 14
NUM_EPOCHS = 25
IMAGE_WIDTH =  1118 # 650 1118
IMAGE_HEIGHT =  1440 # 650 1440
LEARNING_RATE=0.01

# PATH CONSTANTS
SRC_PATH = os.path.dirname(__file__)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, 'dataset', 'Covid19-dataset')
SAVE_PATH = os.path.join(SRC_PATH, 'model', 'saved')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
