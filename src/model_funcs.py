import os
import torch 
import torch.nn as nn
import config
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score

# Define class names explicitly for plotting/reporting
CLASS_NAMES = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", 
               "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", 
               "Cardiomegaly", "Nodule", "Mass", "Hernia"]

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    # Store loss history for plotting
    train_losses = []
    val_losses = []

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
        
        # Calculate Validation Loss
        val_loss = validation_loss()
        
        # Store history
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # Return model AND the loss histories
    return model, train_losses, val_losses

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

def save_model_parameters(model, model_type=None, filename=""):
    """ 
    Save a trained model parameters into appropriate folders based on which model it is. 

    Use a predefined filename

    Argument: 
    - model: Trained PyTorch model
    - model_type: One of config.RESNET, config.IMAGENET, config.CUSTOM_CNN, config.CUSTOM_TRANS
    - filename: String filename to save the parameters as

    Returns: None
    """
    # make sure folders exist
    os.makedirs("parameters/resnet/", exist_ok=True)
    os.makedirs("parameters/imagenet/", exist_ok=True)
    os.makedirs("parameters/custom_cnn/", exist_ok=True)
    os.makedirs("parameters/custom_transformer/", exist_ok=True)

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
    
    full_path = path + filename + ".pth"

    # if file already exists, ask for confirmation to overwrite
    if os.path.exists(full_path):
        response = input(f"File {full_path} already exists. Overwrite? You can choose another filename if n is selected (y/n): ")
        if response.lower() != 'y':
            new_filename = input("Enter new filename (without extension): ")
            full_path = path + new_filename + ".pth"

    torch.save(model.state_dict(), full_path)

from sklearn.metrics import roc_auc_score, accuracy_score

def model_statistics(probs, labels, train_losses=None, val_losses=None):
    """
    Computes detailed statistics and saves plots.
    
    Arguments:
    - probs: Predicted probabilities (numpy array)
    - labels: True labels (numpy array)
    - train_losses: List of training losses per epoch
    - val_losses: List of validation losses per epoch
    """
    # 1. Threshold probabilities to get binary predictions (0 or 1)
    threshold = 0.5
    preds = (probs > threshold).astype(int)

    print("\n" + "="*30)
    print("MODEL EVALUATION REPORT")
    print("="*30)

    # 2. Classification Report (Precision, Recall, F1 per class)
    # zero_division=0 handles classes that might not be present in the test set
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    # 3. ROC AUC Score (Macro average)
    try:
        roc_score = roc_auc_score(labels, probs, average='macro')
        print(f"\nMacro Average ROC AUC: {roc_score:.4f}")
    except ValueError:
        print("\nROC AUC could not be calculated (likely missing positive samples for a class in test set).")

    # 4. Plot Training/Validation Loss Curve
    if train_losses and val_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (BCE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss_curve.png')
        print("Saved loss curve to 'training_loss_curve.png'")
        plt.close()

    # 5. Multi-label Confusion Matrix
    # Since this is multi-label, we get one 2x2 matrix per class. 
    # We will visualize the confusion matrix for the most frequent classes or all of them.
    mcm = multilabel_confusion_matrix(labels, preds)
    
    # Plotting confusion matrices for all 14 classes in a grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20)) # 4x4 grid for 14 classes
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(CLASS_NAMES):
            # cm = [[TN, FP], [FN, TP]]
            sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{CLASS_NAMES[i]}")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_xticklabels(['Neg', 'Pos'])
            ax.set_yticklabels(['Neg', 'Pos'])
        else:
            ax.axis('off') # Hide extra subplots

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion matrices to 'confusion_matrices.png'")
    plt.close()
    
    return preds