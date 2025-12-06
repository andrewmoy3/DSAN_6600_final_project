import config
import torch
import torchvision
import torch.nn as nn


# Define a custom CNN as a class inheriting from nn.Module
class CNN(nn.Module):
    def __init__(self, num_classes): # removed vocab_size/lstm args
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # added stride to reduce size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2) #another max pool to decrease num params
        )
        # Calculate flattened size: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

import torchvision
from torchvision.models import resnet50
from torchvision.models import vit_b_16

# transfer learning
# uses pretrained cnn from torchvision, modifies final layer fc to output num_classes
def make_resnet(num_classes):
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if config.FREEZE_LAYERS == 1:
        # Freeze all layers except the last layer
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.layer4.parameters(): 
            # unfreeze last layer
            param.requires_grad = True 

        for param in model.fc.parameters():
            # unfreeze classifer head
            param.requires_grad = True
    elif config.FREEZE_LAYERS == 2:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            # unfreeze classifer head
            param.requires_grad = True
    return model

def make_vit(num_classes):
    model = vit_b_16(weights="IMAGENET1K_V1")

    if config.FREEZE_LAYERS == 1:
        # Freeze all layers except the last layer
        for param in model.parameters():
            param.requires_grad = False

        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True

        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif config.FREEZE_LAYERS == 2:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.head.parameters():
            param.requires_grad = True

    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    return model