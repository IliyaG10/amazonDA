from unittest import result

import pandas as pd
from pathlib import Path
import csv
#
parent_dir = Path("/media/ghaedi/Data/amazon")
files = list(parent_dir.glob("*.jsonl"))
meta_files = [file for file in files if file.name.startswith("meta_")]
review_files = [file for file in files if not file.name.startswith("meta_")]
with open("meta_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    empty_keys = {key: 0 for key in
                  next(pd.read_json("/media/ghaedi/Data/amazon/meta_Movies_and_TV.jsonl", lines=True, chunksize=1)).columns.tolist()}

    writer.writerow({"filename":"","length":0, **empty_keys}.keys())
    for file in meta_files:
        print(f"{file.name} has started")
        empty_keys = {key: 0 for key in empty_keys}
        length = 0
        for chunk in pd.read_json(file, lines=True, chunksize=10000):
            length += len(chunk)
            for key in empty_keys.keys():
                try:
                    empty_keys[key] += chunk[key].apply(
                        lambda x: (
                            x.strip() == "" if isinstance(x, str)
                            else len(x) == 0 if isinstance(x, (list, dict))
                            else bool(pd.isna(x))
                        )
                    ).sum()

                except KeyError as e:
                    empty_keys[key] = -1
                    continue
        result = list(empty_keys.values())
        result.insert(0, file.name)
        result.insert(1, length)
        writer.writerow(result)
        print(f"{file.name} has finished")
with open("review_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    empty_keys = {key: 0 for key in
                  next(pd.read_json("/media/ghaedi/Data/amazon/Movies_and_TV.jsonl", lines=True, chunksize=1)).columns.tolist()}

    writer.writerow({"filename":"","length":0, **empty_keys}.keys())
    for file in review_files:
        print(f"{file.name} has started")
        empty_keys = {key: 0 for key in empty_keys}
        length = 0
        for chunk in pd.read_json(file, lines=True, chunksize=10000):
            length += len(chunk)
            for key in empty_keys.keys():
                try:
                    empty_keys[key] += chunk[key].apply(
                        lambda x: (
                            x.strip() == "" if isinstance(x, str)
                            else len(x) == 0 if isinstance(x, (list, dict))
                            else bool(pd.isna(x))
                        )
                    ).sum()

                except KeyError as e:
                    empty_keys[key] = -1
                    continue
        result = list(empty_keys.values())
        result.insert(0, file.name)
        result.insert(1, length)
        writer.writerow(result)
        print(f"{file.name} has finished")

