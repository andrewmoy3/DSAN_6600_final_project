import pandas as pd
import os
import random
import config
from torch_funcs import ImageDataset
import torch
from torch.utils.data import DataLoader

from torch_funcs import make_resnet, make_vit
from model_funcs import train_model, evaluate_model, save_model_parameters, load_model_parameters, model_statistics
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images, ids_to_images, get_pos_weights

###############################
# NOTE: When processing data, "Data_Entry_2017_v2020.csv" is treated as the ground truth. It is assumed that the image folders contain all the images listed in the CSV, and no others.
###############################

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

# Choose model based on config.py
if config.MODEL == config.RESNET:
    print("Using ResNet model")
    model = make_resnet(config.NUM_CLASSES)
elif config.MODEL == config.IMAGENET:
    print("Using ImageNet model")
    model = load_model_parameters(make_resnet(config.NUM_CLASSES), config.IMAGENET, "imagenet_model")
elif config.MODEL == config.CUSTOM_CNN:
    print("Using Custom CNN model")
    hidden_size = 128
    num_layers = 2
    # cnn = load_model_parameters(CNN(vocab_size, hidden_size, num_layers), config.CUSTOM_CNN, "custom_cnn_model")  
    pass   
elif config.MODEL == config.CUSTOM_TRANS:
    print("Using Custom Transformer model")
    # to be implemented
    pass

# ----------------- UPDATED TRAINING CALL -----------------
print("Starting Training...")
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
print("Evaluating on Test Set...")
probs, labels = evaluate_model(model, test_loader)

# ----------------- UPDATED STATISTICS CALL -----------------
# Now passing the loss histories
stats = model_statistics(probs, labels, train_losses, val_losses)


