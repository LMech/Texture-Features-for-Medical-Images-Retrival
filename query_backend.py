import pickle

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import consts
import functions


def search_query(img_path: str, imgs_num: int = 10):

    kmeans = pickle.load(open("kmeans.pkl", "rb"))

    evaluated_histogram_list = np.load("evaluated_histogram_list.npy")

    details_file = pd.read_csv("details.csv", header=0)

    img_folders_df = pd.DataFrame(details_file, columns=["path", "class"])

    img = cv2.imread(img_path)

    # Convert to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filter
    filtered_gabor = functions.apply_gabor_filter(img)

    # Apply Schmid filter
    filtered_shmid = functions.apply_schmid_filter(img)

    # Merge the 2 filtered Images
    filtered_img = filtered_gabor + filtered_shmid

    partitioned_images = functions.partition_image(filtered_img)

    classified_image = kmeans.predict(partitioned_images)

    query_histogram, _ = np.histogram(
        classified_image, bins=range(consts.n_clusters + 1), density=True
    )

    similarity_list = []

    for x in tqdm(
        range(evaluated_histogram_list.shape[0]), desc="Calculating similarity"
    ):
        similarity_result = functions.hist_match(
            query_histogram, evaluated_histogram_list[x]
        )
        similarity_list.append(
            [similarity_result, img_folders_df["path"][x], img_folders_df["class"][x]]
        )

    top_image_retrieved = sorted(similarity_list, reverse=True)[:imgs_num]

    return top_image_retrieved
