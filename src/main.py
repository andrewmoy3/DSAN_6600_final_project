import pandas as pd
import os
import random
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images, ids_to_images
from torch_funcs import ImageDataset
import torch
from torch.utils.data import DataLoader

###############################
# NOTE: When processing data, "Data_Entry_2017_v2020.csv" is treated as the ground truth. It is assumed that the image folders contain all the images listed in the CSV, and no others.
###############################

labels_df = get_labels() # DataFrame with all relevant columns

num_image_folders = 1 # Choose how many image folders to process (12 total)
        
num_patients, num_images = get_num_patients_images(num_image_folders)

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
train_data = ImageDataset(ids_to_images(train_patients, labels_df, num_image_folders))
val_data = ImageDataset(ids_to_images(val_patients, labels_df, num_image_folders))
test_data = ImageDataset(ids_to_images(test_patients, labels_df, num_image_folders))

seq_len = 100
batch_size = 640

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

