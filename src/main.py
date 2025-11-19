import pandas as pd
import os
import random
from process_data import get_labels, label_string_to_multi_hot, get_num_patients_images

###############################
# NOTE: When processing data, "Data_Entry_2017_v2020.csv" is treated as the ground truth. It is assumed that the image folders contain all the images listed in the CSV, and no others.
###############################

labels_df = get_labels() # DataFrame with all relevant columns
# labels = labels_df['Finding Labels'].apply(label_string_to_multi_hot) # Take label strings and put in multi-hot encoding

num_image_folders = 1 # Choose how many image folders to process (12 total)
        
num_patients, num_images = get_num_patients_images(num_image_folders)

# split data into train, val, test sets based on number of patients (patient ID)
rand_patient_range = list(range(1, num_patients + 1))
n = len(rand_patient_range) 
assert n == num_patients, "Number of unique patients does not match number of patients from images"
random.shuffle(rand_patient_range)

train_patients = rand_patient_range[0:int(n * 0.7)]
val_patients = rand_patient_range[int(n * 0.7):int(n * 0.85)]
test_patients = rand_patient_range[int(n * 0.85):]

def patient_id_to_folder_num(id):
    for i in range(1, num_image_folders+1):
        folder_path = f'data/images/images_{i:03d}/images'
        image_filenames = pd.Series(os.listdir(folder_path), name='Image Index')
        for filename in image_filenames:
            patient_id = int(filename.split('_')[0])
            if patient_id == id:
                return i
    return 0

# Create datasets of class ImageDataset
# For each patient id, get path to its images