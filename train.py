import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from halo import Halo

import consts
import helpers


def train(descriptor: str):
    spinner = Halo(spinner='dots')

    training_data = helpers.load_preprocessed_data(descriptor=descriptor)
    original_shape = training_data.shape
    training_data = training_data.reshape(
        original_shape[0] * original_shape[1], original_shape[2]
    )

    kmeans = KMeans(
        n_clusters=consts.n_clusters,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=None,
    )
    
    try:
        spinner.start(text='Training KMeans model')
        kmeans.fit(training_data)
        model_path = helpers.create_dirs(consts.models_path, descriptor)
        model_name = f"{descriptor}_kmeans.pkl"
        pkl_model_path = os.path.join(model_path, model_name)
        pickle.dump(kmeans, open(pkl_model_path, "wb"))
        spinner.succeed(f"KMeans model saved as {model_name}")
    except ValueError as e:
        spinner.fail(e)

    imgs_num = original_shape[0]
    evaluated_histogram_list = []

    for i in tqdm(range(imgs_num), desc="Evaluting the training histogram", colour='cyan'):
        evaluted_class = kmeans.predict(
            training_data[
                (i * original_shape[1]) : (i * original_shape[1]) + original_shape[1]
            ]
        )

        img_histogram, _ = np.histogram(
            evaluted_class, bins=range(consts.n_clusters + 1), density=True
        )

        evaluated_histogram_list.append(img_histogram)
    spinner.start(text='Saving evaluated histogram')
    evaluated_histogram_list = np.array(evaluated_histogram_list)
    arr_name = f"{descriptor}_histogram.npy"
    arr_model_path = os.path.join(model_path, arr_name)
    np.save(arr_model_path, evaluated_histogram_list)
    spinner.succeed(f"Evaluted histogram saved as {arr_name}")

    spinner.succeed(f"Model files saved to {os.path.abspath(model_path)}")


train("original")
