import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

import consts
import functions


def preprocess():

    # Create a new details.csv file for each image details
    with open("details.csv", mode="w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=consts.fieldnames)
        writer.writeheader()

    for dataset_class in tqdm(consts.dataset_classes, desc="Preprocessing"):
        class_path = os.path.join(consts.dataset_path, dataset_class)
        class_imgs = os.listdir(class_path)
        for class_img in class_imgs:
            img_path = os.path.join(class_path, class_img)

            img = cv2.imread(img_path)
            # Convert to greyscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gabor filter
            filtered_gabor = functions.apply_gabor_filter(img)

            # Apply Schmid filter
            filtered_shmid = functions.apply_schmid_filter(img)

            # Merge the 2 filtered Images
            filtered_img = filtered_gabor + filtered_shmid

            partitioned_imgs = functions.partition_image(filtered_img)

            img_name = os.path.splitext(class_img)[0]

            output_path = os.path.join(
                consts.preprocessed_dataset_path, dataset_class, f"{img_name}.npy"
            )

            functions.write_csv(
                {
                    "name": class_img,
                    "path": img_path,
                    "preprocessed_path": output_path,
                    "class": dataset_class,
                    "height": img.shape[0],
                    "width": img.shape[1],
                }
            )

            np.save(output_path, partitioned_imgs)


preprocess()
