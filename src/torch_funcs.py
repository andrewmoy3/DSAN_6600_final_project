import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as T
import PIL.Image as Image


# Instantiates all PyTorch related code

# have to define our own dataset class that inherits from PyTorch Dataset
class ImageDataset(Dataset):
    # data = list of tuples (filename, multi-hot label)
    def __init__(self, data):
        self.data = data
        self.transform = T.Compose([
            # T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

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

    # Freeze pretrained parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.fc.parameters():
    #     param.requires_grad = True
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    return model

def make_vit(num_classes):
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model