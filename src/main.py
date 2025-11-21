import pandas as pd
import os
import random
import config
from torch_funcs import ImageDataset
import torch
from torch.utils.data import DataLoader

from torch_funcs import make_resnet, make_vit
from model_funcs import train_model, evaluate_model
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images, ids_to_images

###############################
# NOTE: When processing data, "Data_Entry_2017_v2020.csv" is treated as the ground truth. It is assumed that the image folders contain all the images listed in the CSV, and no others.
###############################

labels_df = get_labels() # DataFrame with all relevant columns

num_patients, num_images = get_num_patients_images(config.NUM_FOLDERS)

print(f"Number of patients: {num_patients}")
print(f"Number of images: {num_images}")

# split data into train, val, test sets based on number of patients (patient ID)
rand_patient_range = list(range(1, num_patients + 1))
n = len(rand_patient_range) 
assert n == num_patients, "Number of unique patients does not match number of patients from images"
random.shuffle(rand_patient_range)

train_patients = rand_patient_range[0:int(n * 0.7)]
val_patients = rand_patient_range[int(n * 0.7):int(n * 0.85)]
test_patients = rand_patient_range[int(n * 0.85):]

# Create datasets of class ImageDataset
# get list of (image_path, multi-hot label) tuples using helper function ids_to_images
train_data = ImageDataset(ids_to_images(train_patients, labels_df, config.NUM_FOLDERS))
val_data = ImageDataset(ids_to_images(val_patients, labels_df, config.NUM_FOLDERS))
test_data = ImageDataset(ids_to_images(test_patients, labels_df, config.NUM_FOLDERS))


train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

# CNN
num_classes = 14  # number of labels of disease
cnn = make_resnet(num_classes)
cnn = train_model(cnn, train_loader, val_loader, num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE)
# save the trained model parameters
torch.save(cnn.state_dict(), 'cnn_model.pth')


probs, labels = evaluate_model(cnn, test_loader)


# Vision Transformer
vit = make_vit(num_classes)
# vit = train_model(vit, train_loader, val_loader, num_epochs=10, lr=1e-4)