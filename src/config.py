# Configurations
MODEL = 0 # Choose model type: 0 - ResNet, 1 - ImageNet, 2 - Custom CNN, 3 - Custom Transformer
MODEL_NAME = "filename" # Filename to save the trained model parameters as
NUM_FOLDERS = 1 # Choose how many image folders to process (12 total)
NUM_CLASSES = 14 # Number of disease classes

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15

# Model types
RESNET = 0
IMAGENET = 1
CUSTOM_CNN = 2
CUSTOM_TRANS = 3
