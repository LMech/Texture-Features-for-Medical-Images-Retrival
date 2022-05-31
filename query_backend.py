from os.path import join
from pickle import load as pkl_load

from cv2 import imread
from numpy import histogram, load
import pandas as pd
from tqdm import tqdm

from consts import models_path, details_path, n_clusters
from helpers import preprocess_img, partition_img, hist_match


def search_query(img_path: str, descriptor: str, imgs_num: int = 10):

    model_path = join(models_path, descriptor)

    pkl_model_path = join(model_path, f"{descriptor}_kmeans.pkl")

    kmeans = pkl_load(open(pkl_model_path, "rb"))

    arr_model_path = join(model_path, f"{descriptor}_histogram.npy")

    evaluated_histogram_list = load(arr_model_path)

    details_file = pd.read_csv(details_path, header=0)

    img_folders_df = pd.DataFrame(details_file, columns=["path", "class"])

    img = imread(img_path)

    filtered_img = preprocess_img(img, descriptor=descriptor)

    partitioned_images = partition_img(filtered_img)

    classified_image = kmeans.predict(partitioned_images)

    query_histogram, _ = histogram(
        classified_image, bins=range(n_clusters + 1), density=True
    )

    similarity_list = []

    for x in tqdm(
        range(evaluated_histogram_list.shape[0]),
        desc="Calculating similarity",
        colour="cyan",
    ):
        similarity_result = hist_match(query_histogram, evaluated_histogram_list[x])
        similarity_list.append(
            [similarity_result, img_folders_df["path"][x], img_folders_df["class"][x]]
        )

    top_image_retrieved = sorted(similarity_list, reverse=True)[:imgs_num]

    return top_image_retrieved
