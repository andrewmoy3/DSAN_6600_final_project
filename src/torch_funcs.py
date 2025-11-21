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
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

# Define a custom CNN as a class inheriting from nn.Module
class CNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # embed character indices
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # feed into LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # feed h nodes into linear layer, output character
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        # feed character embeddings into LSTM
        out, hidden = self.lstm(x.float(), hidden)
        # hidden cell units and lstm cell states at each time step
            # (layers*batches*hidden_size)  
        # out shape (batch_size, seq_len, hidden_size)
        # feed into linear layer
        out = self.fc(out)
        # out shape (batch_size, seq_len, vocab_size)
        return out, hidden

import torchvision
from torchvision.models import resnet50
from torchvision.models import vit_b_16

# transfer learning
# uses pretrained cnn from torchvision, modifies final layer fc to output num_classes
def make_resnet(num_classes):
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_vit(num_classes):
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model