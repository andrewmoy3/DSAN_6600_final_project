import numpy as np
import os
import random
import config
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

from architectures import make_resnet, make_vit, CNN
from model_funcs import train_model, evaluate_model, save_model_parameters, load_model_parameters, model_statistics
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images, ids_to_images, get_pos_weights, ImageDataset


###############################
# NOTE: When processing data, "Data_Entry_2017_v2020.csv" is treated as the ground truth. It is assumed that the image folders contain all the images listed in the CSV, and no others.
###############################
device = "cuda" if torch.cuda.is_available() else "cpu"

labels_df = get_labels() # DataFrame with all relevant columns

num_patients, num_images = get_num_patients_images(config.NUM_FOLDERS)
pos_weight_tensor = get_pos_weights(labels_df)

print(f"Number of patients: {num_patients}")
print(f"Number of images: {num_images}")

#################################################################
# split data into train, val, test sets based on number of patients (patient ID)
rand_patient_range = list(range(1, num_patients + 1))
n = len(rand_patient_range) 
assert n == num_patients, "Number of unique patients does not match number of patients from images"
random.shuffle(rand_patient_range)

train_patients = rand_patient_range[0:int(n * 0.7)]
val_patients = rand_patient_range[int(n * 0.7):int(n * 0.8)]
test_patients = rand_patient_range[int(n * 0.8):]

# Create datasets of class ImageDataset
# get list of (image_path, multi-hot label) tuples using helper function ids_to_images
train_data = ImageDataset(ids_to_images(train_patients, labels_df, config.NUM_FOLDERS))
val_data = ImageDataset(ids_to_images(val_patients, labels_df, config.NUM_FOLDERS))
test_data = ImageDataset(ids_to_images(test_patients, labels_df, config.NUM_FOLDERS))

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)
#################################################################

model = load_model_parameters(config.MODEL, config.MODEL_NAME)
if model is not None:
    model = model.to(device)
else:
    print("Training new model from scratch")
    # Choose model based on config.py
    if config.MODEL == config.RESNET:
        print("Using ResNet model")
        model = make_resnet(config.NUM_CLASSES)
    elif config.MODEL == config.IMAGENET:
        print("Using ImageNet model")
        model = make_vit(config.NUM_CLASSES)    
    elif config.MODEL == config.CUSTOM_CNN:
        print("Using Custom CNN model")
        model = CNN(14)
        pass   
    elif config.MODEL == config.CUSTOM_TRANS:
        print("Using Custom Transformer model")
        # to be implemented
        pass

    # ----------------- TRAINING -----------------
    print("Starting Training")
    model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        pos_weight_tensor,
        num_epochs=config.NUM_EPOCHS, 
        lr=config.LEARNING_RATE
    )

    # Save trained model parameters
    save_model_parameters(model)

# Get predicted probabiities and true labels on test set
print("Evaluating Test Set")
probs, test_labels = evaluate_model(model, test_loader)


# -------------------------- DEFINE BEST THRESHOLDS -----------------------------
val_labels = []
val_probs = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        for row in labels:
            val_labels.append(row.cpu().numpy())
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        outputs = torch.sigmoid(logits) # we apply sigmoid in evaluate_model, so do it here too
        for row in outputs:
            val_probs.append(row.cpu().numpy())

val_labels = np.array(val_labels)
val_probs = np.array(val_probs)
def best_threshold(y_true, y_prob):
    ts = np.linspace(0.01, 0.99, 200)
    f1s = []
    for t in ts:
        preds = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))
    best_idx = np.argmax(f1s)
    return ts[best_idx], f1s[best_idx]
    
thresholds = []
for i in range(14):
    t, f1 = best_threshold(val_labels[:, i], val_probs[:, i])
    print(f"Class {i}: threshold={t:.3f}, val F1={f1:.3f}")
    thresholds.append(t)

thresholds = np.array(thresholds)
print("Best thresholds based on F1 score:")
print(thresholds)
# ------------------------------------------------------------------------------

# -------------------------- PLOTTING CURVES -----------------------------
labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", 
               "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", 
               "Cardiomegaly", "Nodule", "Mass", "Hernia"]


plt.figure(figsize=(8,6))

for i in range(14):
    y_true = val_labels[:, i]
    y_prob = val_probs[:, i]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, label=labels[i])

plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (All 14 Classes)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

for i in range(14):
    y_true = val_labels[:, i]
    y_prob = val_probs[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=labels[i])

plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curves (All 14 Classes)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------

stats = model_statistics(probs, test_labels, thresholds, train_losses, val_losses)


