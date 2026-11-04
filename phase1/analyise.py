import csv
import json
from multiprocessing import Pool
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

parent_dir = Path("/media/ghaedi/Data/amazon")
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)


def is_empty(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (list, dict)):
        return len(x) == 0
    return False


def analyze(file):
    cache = cache_dir / (file.name + ".json")
    if cache.exists():                       # already done in an earlier run
        return json.loads(cache.read_text())

    length = 0
    present = {}   # key -> rows where the key exists
    empty = {}     # key -> rows where the key exists but is empty

    with open(file, "rb") as f:              # read one line at a time
        for line in f:
            if not line.strip():
                continue
            row = loads(line)
            length += 1
            for key, value in row.items():
                present[key] = present.get(key, 0) + 1
                if is_empty(value):
                    empty[key] = empty.get(key, 0) + 1
            if length % 1_000_000 == 0:
                print(f"{file.name}: {length:,} rows done", flush=True)

    result = {"filename": file.name, "length": length,
              "present": present, "empty": empty}
    cache.write_text(json.dumps(result))
    print(f"{file.name} finished", flush=True)
    return result


def write_reports(results, prefix):
    all_keys = []
    for r in results:
        for k in r["present"]:
            if k not in all_keys:
                all_keys.append(k)

    # 1) number of rows where each field is missing or empty
    with open(f"{prefix}_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "length", *all_keys])
        for r in results:
            row = []
            for k in all_keys:
                p = r["present"].get(k, 0)
                row.append((r["length"] - p) + r["empty"].get(k, 0))
            w.writerow([r["filename"], r["length"], *row])

    # 2) number of rows where each key exists at all
    with open(f"{prefix}_presence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "length", *all_keys])
        for r in results:
            w.writerow([r["filename"], r["length"],
                        *[r["present"].get(k, 0) for k in all_keys]])

    # 3) which keys are not in every file
    key_sets = [set(r["present"]) for r in results]
    common = set.intersection(*key_sets)
    for k in all_keys:
        if k not in common:
            missing = [r["filename"] for r in results if k not in r["present"]]
            print(f"[{prefix}] '{k}' is missing in: {missing}")


if __name__ == "__main__":
    files = sorted(parent_dir.glob("*.jsonl"))
    meta_files = [f for f in files if f.name.startswith("meta_")]
    review_files = [f for f in files if not f.name.startswith("meta_")]

    with Pool(processes=3) as pool:          # 2-4 is enough; the disk is the limit
        meta_results = pool.map(analyze, meta_files)
        review_results = pool.map(analyze, review_files)

    write_reports(meta_results, "meta")
    write_reports(review_results, "review")