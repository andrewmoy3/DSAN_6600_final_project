# Deep Learning on Chest X-Ray Images for Classification of Thoracic Diseases

**Authors:** Andrew Moy, Samuel Villarreal & Xiao Xu
**Course:** DSAN-6600 — Fall 2025  
**Instructors:** Dr. Hickman & Benjamin Houghton  
**Date:** December 8, 2025

---

## **Introduction**

With the rapid advancement of technology, the healthcare industry has been able to capture and analyze more data than ever before. Among imaging modalities, X-rays remain one of the most widely used diagnostic tools, utilizing high-energy electromagnetic radiation that penetrates tissue at varying degrees. This allows clinicians to visualize internal anatomical structures, particularly dense regions such as bone, and identify conditions like fractures, infections, and pulmonary abnormalities without invasive procedures.

Modern hospitals nowadays collect large thoracic imaging data. With the progress of machine learning and deep learning architectures, these datasets can be leveraged to automate the detection and classification of thoracic diseases, significantly improving efficiency and faster clinical decision-making.

This project analyzesa total of 108,948 frontal-view chest X-ray images from 32,717 unique patients, provided by the U.S. National Institutes of Health (NIH), to classify the following 14 common thoracic findings:

1. Atelectasis
2. Consolidation
3. Edema
4. Emphysema
5. Fibrosis
6. Pleural Thickening
7. Cardiomegaly
8. Effusion
9. Infiltration
10. Mass
11. Nodule
12. Hernia
13. Pneumonia
14. Pneumothorax

For reference, Pneumonia alone accounted for 41,210 deaths in the United States in 2023, corresponding to a mortality rate of 12.3 deaths per 100,000 people, according to the Centers for Disease Control and Prevention.

---

## **Project Objective**

The goal of this project is to develop a well-documented Python framework that applies advanced deep learning models to accurately classify the aforementioned thoracic conditions. We evaluate and compare Convolutional Neural Networks (CNNs) and Transformer-based architectures to determine the most effective model for this task.

---

## **Code Overview**

The following is a brief overview of the main code files and the descriptions of their functions. More detailed explanations can be found below each function's definition.

### architectures.py

This file contains two function definitions and a custom PyTorch neural network class for the creation of PyTorch deep learning models:

    1. `make_resnet(num_classes)`: Constructs a ResNet-50 model pre-trained on ImageNet, modifying its final dense layer to output the specified number of classes.

    2. `make_vit(num_classes)`: Builds a Vision Transformer (ViT) model, pre-trained on ImageNet, tailored for the specified number of classes.

    3. `CustomCNN`: A custom convolutional neural network class designed for this project.

### config.py

This file contains configuration variables used throughout all other code files, and allows the user to easily modify hyperparameters and settings. It allows the user to construct the model without the need to modify other code files. The options available for customization are as follows:

    1. MODEL # Choose model type: 0 - ResNet, 1 - ImageNet, 2 - Custom CNN, 3 - Custom Transformer

    2. MODEL_NAME # Filename to save the trained model parameters as

    3. NUM_FOLDERS # Choose how many image folders to process (12 total)

    4. NUM_CLASSES # Number of disease classes

    5. DATA_AUGMENT # Whether to apply data augmentation during training

    6. FREEZE_LAYERS # 0 - no freezing, 1 - freeze all but last layer, 2 - freeze all but last two layers

    7. BATCH_SIZE # Batch size for training

    8. LEARNING_RATE # Learning rate for optimizer

    9. NUM_EPOCHS # Number of training epochs

    10. WEIGHT_DECAY # Weight decay for optimizer

### main.py

This is the file that manages the overall flow of the program. It is the only file that should be executed, and oversees the execution of all other modules/functions, including data processing, training, evaluation, and reporting.

It first separates the dataset into training, validation, and test sets. It then chooses the correct deep learning model based on the user's configuration settings, and either trains the model from scratch or loads pre-trained parameters. Finally, it evaluates the model's performance on the test set and generates statistics and plots to describe the results.

### model_funcs.py

This file contains various functions used for model training, evaluation, saving/loading, and reporting. The functions included are:

    1. `train_model(...)`: Trains a given PyTorch model using the provided training datasets and hyperparameters.

    2. `evaluate_model(...)`: Evaluates the trained model on the test dataset and returns predicted probabilities and true labels.

    3. `load_model_parameters(...)`: Loads and returns pre-trained model parameters from the 'parameters/{model_type}' directory.

    4. `save_model_parameters(...)`: Saves a trained model's parameters into the 'parameters/{model_type}' directory.

    5. `model_statistics(...)`: Generates, saves and prints a report of the model's performance on the test dataset.

### preprocess.py

This file provides functions for preprocessing the raw chest X-ray images. It is designed to be single use only, and should only be executed once after downloading the raw data.

    1. `resize_images_to_rgb()`: Resizes all images to size 224x224 and converts them to RGB format.

    2. `augment_data()`: Doubles the dataset size by flipping every image along the y axis.

### process_data.py

This file contains various functions for loading and splitting the images and defines a PyTorch Dataset class. The functions included are:

    1. `get_labels()`: Loads the CSV file mapping image IDs to patient IDs and disease labels.

    2. `label_string_to_multi_hot(s)`: Converts a string of disease labels into a multi-hot encoded vector of length 14

    3. `get_num_patients_images(num_image_folders)`: Returns the number of unique patients and images based on how many image folders to process.

    4. `ids_to_images(ids, labels_df, num_image_folders)`: Returns the image paths and multi-hot labels for a list of image IDs.

    5. `get_pos_weights()`: Calculates and returns the positive weights for each disease class to address class imbalance.
