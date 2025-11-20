import pandas as pd
import time
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

def ids_to_images(ids, labels_df, num_image_folders):
    start = time.perf_counter()
    folders = []
    for i in range(1, num_image_folders+1):
        folder_path = f'data/images/images_{i:03d}/images'
        image_filenames = pd.Series(os.listdir(folder_path), name='Image Index')
        folder_ids = set()
        for filename in image_filenames:
            patient_id = int(filename.split('_')[0])
            folder_ids.add(patient_id)
        folders.append(folder_ids)

    def patient_id_to_folder_number(id):
        for i, id_set in enumerate(folders, start=1):
            if patient_id in id_set:
                return i
        return 0

    # take labels_df, create a dict mapping id to number of images
    id_to_num_images = labels_df['Patient ID'].value_counts().to_dict()

    # map image index to finding labels string
    image_index_to_label_str = labels_df.set_index('Image Index')['Finding Labels'].to_dict()

    data = []
    for p_id in ids:
        folder_num = patient_id_to_folder_number(p_id)
        if folder_num == 0:
            continue
        folder_path = f'data/images/images_{folder_num:03d}/images'
        num_images = id_to_num_images.get(p_id, 0)
        if num_images == 0:
            continue
        for img_idx in range(num_images):
            img_fn = f"{p_id:08d}_{img_idx:03d}.png"
            img_path = os.path.join(folder_path, img_fn)
            label_str = image_index_to_label_str.get(img_fn, "")
            label = label_string_to_multi_hot(label_str)
            data.append((img_path, label))
    
    end = time.perf_counter()
    # print(f"time = {end - start}")
    return data

    