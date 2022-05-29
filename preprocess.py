import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

import consts
import functions


def preprocess():
    with open('details.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=consts.fieldnames)        
        writer.writeheader()

    for dataset_class in tqdm(consts.dataset_classes, desc='Preprocessing'):
        class_path = os.path.join(consts.dataset_path, dataset_class)
        class_imgs = os.listdir(class_path)
        for class_img_path in class_imgs:
            img_path = os.path.join(class_path, class_img_path)
            img = cv2.imread(img_path)
            # Convert to greyscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gabor filter
            filtered_gabor = functions.apply_gabor_filter(img)
            
            # Apply Schmid filter
            filtered_shmid = functions.apply_schmid_filter(img)
            
            # Merge the 2 filtered Images
            filtered_img = filtered_gabor + filtered_shmid
            
            partitioned_images = functions.partition_image(filtered_img, consts.partition_patch, consts.partition_patch)
            
            img_name = os.path.splitext(class_img_path)[0]
            output_path = os.path.join(consts.preprocessed_dataset_path, dataset_class, img_name)            
            functions.write_csv({'name': class_img_path, 'path': img_path, 'preprocessed_path': output_path,'class': dataset_class,'height': img.shape[0], 'width': img.shape[1]})
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
            for i in range(len(partitioned_images)):
                partation_name = f'{i}.npy'
                array_path = os.path.join(output_path, partation_name)
                np.save(array_path, partitioned_images[i])


preprocess()
