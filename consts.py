import os

dataset_path = "dataset"
preprocessed_dataset_path = "preprocessed_dataset"
dataset_classes = os.listdir(dataset_path)
models_path = 'models'

n_clusters = 200

fieldnames = ["name", "class", "path", "height", 'width',]
