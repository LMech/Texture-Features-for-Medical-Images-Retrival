from os import listdir
from os.path import join

from pandas import read_csv

dataset_path = "dataset"
preprocessed_dataset_path = "preprocessed_dataset"
dataset_classes = listdir(dataset_path)
models_path = "models"
details_path = join(dataset_path, "details.csv")

details_df = read_csv(
    details_path,
    usecols=["name", "path", "class"],
    dtype={"name": str, "class": str},
)

n_clusters = 200
