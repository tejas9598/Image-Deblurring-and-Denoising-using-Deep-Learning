import torch
import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Model parameters
MODEL_TYPE = "unet"  # Options: "unet" or "dncnn"
IN_CHANNELS = 3  # RGB images
OUT_CHANNELS = 3
NUM_FILTERS = 64

# Image degradation parameters
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 1.0
NOISE_STD = 0.1

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training/validation split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Image preprocessing
IMAGE_SIZE = 128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Checkpoint and logging
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
SAVE_FREQUENCY = 5

# Evaluation metrics
METRICS = ['psnr', 'ssim'] 