# Configurations
MODEL = 0 # Choose model type: 0 - ResNet, 1 - ImageNet, 2 - Custom CNN, 3 - Custom Transformer
MODEL_NAME = "12_folder_alpha_1e-4_10_epochs" # Filename to save the trained model parameters as
NUM_FOLDERS = 12 # Choose how many image folders to process (12 total)
NUM_CLASSES = 14 # Number of disease classes

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4

# Model types
RESNET = 0
IMAGENET = 1
CUSTOM_CNN = 2
CUSTOM_TRANS = 3
