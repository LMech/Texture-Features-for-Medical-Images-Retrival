import os

import numpy as np

schmid_kernel_size = 49
gabor_kernel_size = (15, 15)

schmid_filter_parameter_values = [
    [2, 1],
    [4, 2],
    [6, 2],
    [8, 2],
    [10, 2],
]

gabor_filter_parameter_values = [
    [2, 0, 4, .3,],
    [1, 0, 4, .4,],
    [2, np.pi/2, 8, .3,],
    [1, np.pi/2, 8, .4,],
]

partition_patch = 64

dataset_path = 'dataset'
preprocessed_dataset_path = 'preprocessed_dataset'
dataset_classes = os.listdir(dataset_path)

n_clusters = 200

fieldnames = ['name', 'path', 'preprocessed_path','class','height','width']


