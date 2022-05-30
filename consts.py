import os

dataset_path = "dataset"
preprocessed_dataset_path = "preprocessed_dataset"
dataset_classes = os.listdir(dataset_path)

n_clusters = 200

fieldnames = ["name", "path", "preprocessed_path", "class", "height", "width"]
