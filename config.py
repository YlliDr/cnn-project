# Training settings

IMAGE_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 16

LEARNING_RATE = 1e-3

PATIENCE = 8

SEED = 42


# Dataset paths

IMAGE_DIR = "data/images"

MASK_DIR = "data/masks"


# Output directories

MODEL_DIR = "outputs/models"

LOG_DIR = "outputs/logs"

PLOT_DIR = "outputs/plots"

PREDICTION_DIR = "outputs/predictions"

DEMO_OUTPUT_DIR = "outputs/demo_results"


# Model paths

MODEL_PATH = "outputs/models/best_sod_model.pth"

CHECKPOINT_PATH = "outputs/models/last_checkpoint.pth"