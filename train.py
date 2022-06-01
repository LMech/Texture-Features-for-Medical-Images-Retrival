from os.path import abspath, join
from pickle import dump

from halo import Halo
from numpy import array, histogram, save
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.consts import models_path, n_clusters
from utils.helpers import create_dirs, load_preprocessed_data


def train(descriptor: str):
    spinner = Halo(spinner="dots")

    training_data = load_preprocessed_data(descriptor=descriptor)
    original_shape = training_data.shape
    training_data = training_data.reshape(
        original_shape[0] * original_shape[1], original_shape[2]
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=None,
    )

    try:
        spinner.start(text="Training KMeans model")
        kmeans.fit(training_data)
        model_path = create_dirs(models_path, descriptor)
        model_name = f"{descriptor}_kmeans.pkl"
        pkl_model_path = join(model_path, model_name)
        dump(kmeans, open(pkl_model_path, "wb"))
        spinner.succeed(f"KMeans model saved as {model_name}")
    except ValueError as e:
        spinner.fail(e)

    imgs_num = original_shape[0]
    evaluated_histogram_list = []

    for i in tqdm(
        range(imgs_num), desc="Evaluting the training histogram", colour="cyan"
    ):
        evaluted_class = kmeans.predict(
            training_data[
                (i * original_shape[1]) : (i * original_shape[1]) + original_shape[1]
            ]
        )

        img_histogram, _ = histogram(
            evaluted_class, bins=range(n_clusters + 1), density=True
        )

        evaluated_histogram_list.append(img_histogram)
    spinner.start(text="Saving evaluated histogram")
    evaluated_histogram_list = array(evaluated_histogram_list)
    arr_name = f"{descriptor}_histogram.npy"
    arr_model_path = join(model_path, arr_name)
    save(arr_model_path, evaluated_histogram_list)
    spinner.succeed(f"Evaluted histogram saved as {arr_name}")

    spinner.succeed(f"Model files saved to {abspath(model_path)}")


train("original")
