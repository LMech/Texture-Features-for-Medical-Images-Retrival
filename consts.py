import os

dataset_path = "dataset"
preprocessed_dataset_path = "preprocessed_dataset"
dataset_classes = os.listdir(dataset_path)
models_path = "models"
details_path = os.path.join(dataset_path, "details.csv")

n_clusters = 200
