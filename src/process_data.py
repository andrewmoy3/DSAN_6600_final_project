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
    Filters the main dataframe for specific patient IDs and returns the valid paths.
    """
    # Filter DF for these patients
    subset_df = labels_df[labels_df['Patient ID'].isin(ids)]
    
    # Create a mapping of Image Index -> Folder Path
    # (You only need to build this map once, effectively)
    image_path_map = {}
    for i in range(1, num_image_folders + 1):
        folder_path = f'data/images/images_{i:03d}/images'
        # Only map images that actually exist in the folders we are using
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                image_path_map[img_name] = os.path.join(folder_path, img_name)

    data = []
    # Iterate through the dataframe rows
    for _, row in subset_df.iterrows():
        img_name = row['Image Index']
        label_str = row['Finding Labels']
        
        # Only add if the image exists in the folders we selected
        if img_name in image_path_map:
            full_path = image_path_map[img_name]
            label = label_string_to_multi_hot(label_str)
            data.append((full_path, label))
            
    return data

    