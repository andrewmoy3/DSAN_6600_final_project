# Configurations
MODEL = 0 # Choose model type: 0 - ResNet, 1 - ImageNet, 2 - Custom CNN, 3 - Custom Transformer
MODEL_NAME = "data_aug_2" # Filename to save the trained model parameters as
NUM_FOLDERS = 12 # Choose how many image folders to process (12 total)
NUM_CLASSES = 14 # Number of disease classes
DATA_AUGMENT = True # Whether to apply data augmentation during training
FREEZE_LAYERS = 0 # 0 - no freezing, 1 - freeze all but last layer, 2 - freeze all 

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-6
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-3

# Model types
RESNET = 0
IMAGENET = 1
CUSTOM_CNN = 2
CUSTOM_TRANS = 3
