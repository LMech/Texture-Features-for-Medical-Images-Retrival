import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import consts
import functions


def train(method: str = "original"):
    training_data = functions.load_preprocessed_data(method=method)
    init = "k-means++"
    max_iter = 300
    n_init = 10

    kmeans = KMeans(
        n_clusters=consts.n_clusters,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=None,
    )
    
    try:
        print("training KMeans...")
        kmeans.fit(training_data)
    except ValueError as e:
        print(e)

    model_path = functions.create_dirs(consts.models_path, method)

    model_name = f"{method}_kmeans.pkl"

    print(f"Saving KMeans model as {model_name}")

    pkl_model_path = os.path.join(model_path, model_name)

    pickle.dump(kmeans, open(pkl_model_path, "wb"))
    imgs_num = training_data.shape[0] // training_data.shape[1]
    evaluated_histogram_list = []

    for i in tqdm(range(imgs_num), desc="Evaluting the training histogram"):
        evaluted_class = kmeans.predict(
            training_data[
                (i * training_data.shape[1]) : (i * training_data.shape[1])
                + training_data.shape[1]
            ]
        )

        img_histogram, _ = np.histogram(
            evaluted_class, bins=range(consts.n_clusters + 1), density=True
        )

        evaluated_histogram_list.append(img_histogram)

    evaluated_histogram_list = np.array(evaluated_histogram_list)
    
    arr_name = f"{method}_histogram.npy"
    
    arr_model_path = os.path.join(model_path, arr_name)
    
    np.save(arr_model_path, evaluated_histogram_list)


train()
