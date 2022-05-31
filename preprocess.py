import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import consts
import helpers


def preprocess(descriptor: str):
    details_df = pd.read_csv('details.csv')

    for dataset_class in consts.dataset_classes:
        helpers.create_dirs(
            consts.preprocessed_dataset_path,
            descriptor,
            dataset_class,
        )

    for i in tqdm(range(len(details_df)), desc=f'Preprocessing using {descriptor} descriptor'):

        img = cv2.imread(details_df.at[i, 'path'])

        filtered_imgs = helpers.preprocess_image(img, descriptor)
        
        partitioned_imgs = helpers.partition_image(filtered_imgs)
        
        img_name = os.path.splitext(details_df.at[i, 'name'])[0]

        ary_output_path = os.path.join(
            consts.preprocessed_dataset_path,
            descriptor,
            str(details_df.at[i, 'class']),
            f'{img_name}.npy',
        )
        details_df.at[i, descriptor] = ary_output_path
        np.save(ary_output_path, partitioned_imgs)
    details_df.to_csv('details.csv')


preprocess('hog')
