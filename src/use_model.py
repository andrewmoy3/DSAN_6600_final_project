import pandas as pd
import os
import random
import config
from torch_funcs import ImageDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from torch_funcs import make_resnet, make_vit
from model_funcs import train_model, evaluate_model, save_model_parameters, load_model_parameters, model_statistics
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images, ids_to_images, get_pos_weights

device = "cuda" if torch.cuda.is_available() else "cpu"

labels_df = get_labels() # DataFrame with all relevant columns

num_patients, num_images = get_num_patients_images(config.NUM_FOLDERS)
pos_weight_tensor = get_pos_weights(labels_df)

print(f"Number of patients: {num_patients}")
print(f"Number of images: {num_images}")

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

start = time.time()
model = load_model_parameters(config.MODEL, config.MODEL_NAME)
model = model.to(device)
end = time.time()
print(f"Model loaded in {end - start:.2f} seconds")

probs, test_labels = evaluate_model(model, test_loader)

val_labels = []
val_probs = []    

from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

start = time.time()
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

end = time.time()
print(f"Validation predictions generated in {end - start:.2f} seconds")

for i in range(14):
    print(i)
    y_true = val_labels[:, i]
    y_prob = val_probs[:, i]

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve – Class {i}")
    plt.grid(True)
    plt.show()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve – Class {i}")
    plt.grid(True)
    plt.show()

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
stats = model_statistics(probs, test_labels, thresholds, save=False)