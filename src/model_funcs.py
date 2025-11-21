import torch 
import torch.nn as nn
import config
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    def validation_loss():
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        return val_loss

    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()     # multi-label
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}")

        # val_loss = validation_loss()
        # print(f"Validation Loss: {val_loss:.4f}")

    return model

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test dataset and returns predicted probabilities and true labels.

    Arguments:
    - model: Trained PyTorch model
    - test_loader: DataLoader for the test dataset 

    Returns: Tuple (all_probs, all_labels)
    - all_probs: Numpy array (length 14) of predicted probabilities for each disease type
    - all_labels: Numpy array of true multi-hot encoded labels
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs) # 14 raw scores, 1 for each disease
            probs = torch.sigmoid(logits) # convert to probabilities between 0 and 1

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_probs, all_labels

def load_model_parameters(model, model_type, filename):
    """ 
    Load pre-trained model parameters into the given model based on which model it is. 

    Use a predefined filename

    Argument: 
    - model: Untrained PyTorch model
    - model_type: One of config.RESNET, config.IMAGENET, config.CUSTOM_CNN, config.CUSTOM_TRANS
    - filename: String filename to load the parameters from 
    Returns: Pytorch model with loaded parameters
    """
    if model_type == config.RESNET:
        path = "parameters/resnet/"
    elif model_type == config.IMAGENET:
        path = "parameters/imagenet/"
    elif model_type == config.CUSTOM_CNN:
        path = "parameters/custom_cnn/"
    elif model_type == config.CUSTOM_TRANS:
        path = "parameters/custom_transformer/"
    else:
        raise ValueError("Model type is not recognized")

    model.load_state_dict(torch.load(path + filename + ".pth"))
    return model

def save_model_parameters(model, model_type, filename):
    """ 
    Save a trained model parameters into appropriate folders based on which model it is. 

    Use a predefined filename

    Argument: 
    - model: Trained PyTorch model
    - model_type: One of config.RESNET, config.IMAGENET, config.CUSTOM_CNN, config.CUSTOM_TRANS
    - filename: String filename to save the parameters as

    Returns: None
    """
    if model_type == config.RESNET:
        path = "parameters/resnet/"
    elif model_type == config.IMAGENET:
        path = "parameters/imagenet/"
    elif model_type == config.CUSTOM_CNN:
        path = "parameters/custom_cnn/"
    elif model_type == config.CUSTOM_TRANS:
        path = "parameters/custom_transformer/"
    else:
        raise ValueError("Model type is not recognized")
    
    filename = filename

    torch.save(model.state_dict(), path + filename + ".pth")