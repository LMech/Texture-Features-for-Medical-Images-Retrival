import sys
from os.path import join

import pandas as pd

sys.path.append("../dss_project")
from consts import classes_cnt, details_df, models_path


def calculate_metrics(descriptor: str):
    file_name = f"{descriptor}_evaluation.csv"
    file_path = join(models_path, descriptor, file_name)
    labels_df = pd.read_csv(file_path)
    arp = round((100 / len(labels_df)) * labels_df["percision"].sum(), 2)
    arr = round((100 / len(labels_df)) * labels_df["recall"].sum(), 2)

    fscore = round((2 * arp * arr) / (arp + arr), 2)

    data = [
        ["Percision Average", round(labels_df["percision"].mean() * 100, 2)],
        ["Recall Average", round(labels_df["recall"].mean() * 100, 2)],
        ["ARP", arp],
        ["ARR", arr],
        ["FScore", fscore],
    ]
    print(descriptor)
    print(pd.DataFrame(data))

calculate_metrics("original")
calculate_metrics("hog")
