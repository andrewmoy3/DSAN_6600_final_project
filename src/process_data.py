import pandas as pd
import time
import os

def get_labels():
    """
    Reads the CSV file containing the image labels and image names.

    Returns: Pandas DataFrame 
    """
    filepath = 'data/Data_Entry_2017_v2020.csv'
    df = pd.read_csv(filepath)

    # Cols = Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Sex, View Position, OriginalImage[Width Height], OriginalImagePixelSpacing[x y]
    cols_to_keep = ['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Sex', 'View Position']
    df = df[cols_to_keep] # Remove unnecessary columns

    df = df.dropna(subset=['Finding Labels']) # Drop rows with no labels
    return df

def label_string_to_multi_hot(s):
    """
    Converts a single string of labels separated by '|' into a multi-hot encoded list.

    Argument: String of labels separated by '|'

    Returns: Multi-hot encoded list representing the presence of each label
    """
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
    """
    Counts the number of unique patients and total images in the specified number of image folders. While the CSV file is treated as ground truth, this function counts images directly from the folders because we do not always want to process all 12 folders, and the CSV does not specify which folder an image is in.

    Argument: Number of image folders to process 

    Returns: Tuple (number of unique patients, total number of images)
    """
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
    """
    Maps patient IDs to their corresponding image file paths and multi-hot encoded labels.

    Arguments:
    - ids: List of patient IDs
    - labels_df: DataFrame containing image labels and metadata
    - num_image_folders: Number of image folders to process

    Returns: List of tuples corresponding to each patient id (image_path, multi-hot label of disease in that image)
    """
    
    # Create a list of length 'num_image_folders' 
    # Each element is a set of the patient IDs in that folder
    folders = []
    for i in range(1, num_image_folders+1):
        folder_path = f'data/images/images_{i:03d}/images'
        image_filenames = pd.Series(os.listdir(folder_path), name='Image Index')
        folder_ids = set()
        for filename in image_filenames:
            patient_id = int(filename.split('_')[0])
            folder_ids.add(patient_id)
        folders.append(folder_ids)

    # Helper function to quickly look up which folder a patient ID is in
    def patient_id_to_folder_number(id):
        for i, id_set in enumerate(folders, start=1):
            if id in id_set:
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
    
    return data

    