import csv
import sys
from os.path import join

from tqdm import tqdm

sys.path.append("../dss_project")
from utils.consts import classes_cnt, details_df, models_path
from query_backend import search_query


def generate_labels(descriptor: str):
    file_name = f"{descriptor}_evaluation.csv"
    file_path = join(models_path, descriptor, file_name)
    with open(file_path, mode="w") as csv_file:
        fieldnames = [
            "id",
            "label",
            *[str(i + 1) for i in range(10)],
            "matches_cnt",
            "percision",
            "recall",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in tqdm(
            details_df.iterrows(),
            desc=f"Labeling the data using the {descriptor} descriptor",
            colour="cyan",
            total=len(details_df),
        ):
            row_map = {"id": row["name"], "label": row["class"]}
            results = search_query(row["path"], descriptor, 11)[1:]
            for i, result in enumerate(results):
                row_map.update({str(i + 1): result[2]})
                matched_labels = [int(result[2]) for result in results].count(
                    int(row["class"])
                )
            percision = matched_labels / len(results)
            recall = matched_labels / classes_cnt[row["class"]]
            row_map.update(
                {
                    "matches_cnt": matched_labels,
                    "percision": percision,
                    "recall": recall,
                }
            )
            writer.writerow(row_map)


generate_labels("original")
