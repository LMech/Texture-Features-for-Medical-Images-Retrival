from argparse import ArgumentParser
from os.path import abspath, join

from cv2 import imread
from halo import Halo
from numpy import save
from tqdm import tqdm

from utils.consts import dataset_classes, details_df, preprocessed_dataset_path
from utils.helpers import create_dirs, partition_img, preprocess_img


def preprocess(descriptor: str):

    descriptor_path = join(
        preprocessed_dataset_path,
        descriptor,
    )

    for dataset_class in dataset_classes:
        create_dirs(
            descriptor_path,
            dataset_class,
        )

    for _, row in tqdm(
        details_df.iterrows(),
        desc=f"Preprocessing using the {descriptor} descriptor",
        colour="cyan",
        total=len(details_df),
    ):

        img = imread(row["path"])

        filtered_imgs = preprocess_img(img, descriptor)

        partitioned_imgs = partition_img(filtered_imgs)

        name = row["name"]

        ary_output_path = join(
            descriptor_path,
            row["class"],
            f"{name}.npy",
        )

        save(ary_output_path, partitioned_imgs)

    spinner = Halo(spinner="dots")
    spinner.succeed(
        f"The preprocessed data successfully saved to {abspath(descriptor_path)}"
    )


parser = ArgumentParser()
parser.add_argument(
    "descriptor", type=str, help="Specifity the descriptor to work with"
)
descriptor = parser.parse_args().descriptor
preprocess(descriptor)
