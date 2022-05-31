import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from halo import Halo

import consts
import helpers


def preprocess(descriptor: str):
    details_df = pd.read_csv(consts.details_path)

    descriptor_path = os.path.join(
        consts.preprocessed_dataset_path,
        descriptor,
    )

    for dataset_class in consts.dataset_classes:
        helpers.create_dirs(
            descriptor_path,
            dataset_class,
        )

    for i in tqdm(
        range(len(details_df)), desc=f"Preprocessing using the {descriptor} descriptor", colour='cyan'
    ):

        img = cv2.imread(details_df.at[i, "path"])

        filtered_imgs = helpers.preprocess_image(img, descriptor)

        partitioned_imgs = helpers.partition_image(filtered_imgs)

        name = str(details_df.at[i, "name"])

        ary_output_path = os.path.join(
            descriptor_path,
            str(details_df.at[i, "class"]),
            f"{name}.npy",
        )

        np.save(ary_output_path, partitioned_imgs)
        
    spinner = Halo(spinner='dots')
    spinner.succeed(
        f"The preprocessed data successfully saved to {os.path.abspath(descriptor_path)}"
    )


preprocess("original")
