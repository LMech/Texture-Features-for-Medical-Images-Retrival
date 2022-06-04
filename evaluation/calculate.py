import csv
from os.path import dirname, join
from sys import path

import matplotlib.pyplot as plt
import pandas as pd

path.append(dirname(path[0]))
from utils.consts import models_path


def calculate_metrics(descriptors: list[str]):
    file_path = join("evaluation", "results.csv")
    with open(file_path, mode="w") as csv_file:
        fieldnames = [
            "descriptor",
            "arp",
            "arr",
            "fscore",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for descriptor in descriptors:
            evaluation_file_name = f"{descriptor}_evaluation.csv"
            evaluation_file_path = join(models_path, descriptor, evaluation_file_name)
            labels_df = pd.read_csv(evaluation_file_path)
            arp = round((100 / len(labels_df)) * labels_df["percision"].sum(), 2)
            arr = round((100 / len(labels_df)) * labels_df["recall"].sum(), 2)

            fscore = round((2 * arp * arr) / (arp + arr), 2)

            row_map = {
                "descriptor": descriptor,
                "arp": arp,
                "arr": arr,
                "fscore": fscore,
            }
            writer.writerow(row_map)
    results_df = pd.read_csv(file_path)
    results_df.plot.bar(x="descriptor")
    plt.title("Results Bar Plot")
    plt.xlabel("Decriptor")
    plt.ylabel("Percentage %")
    plt.grid(True, linestyle="--", axis="y")
    plt.xticks(rotation=0, ha='right')
    plt.savefig("evaluation/calculated_metrics.svg")
    plt.show()


calculate_metrics(["original", "hog"])
