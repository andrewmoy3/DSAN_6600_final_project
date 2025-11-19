import pandas as pd
import os

def get_labels():
    filepath = 'data/Data_Entry_2017_v2020.csv'
    df = pd.read_csv(filepath)

    # Cols = Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Sex, View Position, OriginalImage[Width Height], OriginalImagePixelSpacing[x y]
    cols_to_keep = ['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Sex', 'View Position']
    df = df[cols_to_keep] # Remove unnecessary columns

    df = df.dropna(subset=['Finding Labels']) # Drop rows with no labels
    return df

def label_string_to_multi_hot(s):
    labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
    label_dict = {label: i for i, label in enumerate(labels)}

    s = s.split('|')
    multi_class = [0] * len(labels)
    for label in s:
        if label in label_dict:
            multi_class[label_dict[label]] = 1
        elif label == 'No Finding':
            continue
        else:
            print(f"Unknown label: {label}")
    return multi_class

def get_num_patients_images(num_image_folders):
    # get the number of patients and images from the filenames
    # necessary because Data_Entry_2017_v2020.csv does not divide by folders
    num_patients = 0
    num_images = 0

    for i in range(1, num_image_folders+1):
        folder_path = f'data/images/images_{i:03d}/images'
        image_filenames = pd.Series(os.listdir(folder_path), name='Image Index')
        num_images += len(image_filenames)
        for filename in image_filenames:
            patient_id = filename.split('_')[0]
            if int(patient_id) > num_patients:
                num_patients = int(patient_id)

    return num_patients, num_images
