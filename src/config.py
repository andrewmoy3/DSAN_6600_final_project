# Configurations
MODEL = 0 # Choose model type: 0 - ResNet, 1 - ImageNet, 2 - Custom CNN, 3 - Custom Transformer
MODEL_NAME = "focal_loss_10_folders_1e-7_high_decay" # Filename to save the trained model parameters as
NUM_FOLDERS = 10 # Choose how many image folders to process (12 total)
NUM_CLASSES = 14 # Number of disease classes

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-7
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-3

# Model types
RESNET = 0
IMAGENET = 1
CUSTOM_CNN = 2
CUSTOM_TRANS = 3
