
import csv
import json
import math
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads


parent_dir = Path("/media/ghaedi/Data/amazon")
cache_dir = Path("quality_cache")
cache_dir.mkdir(exist_ok=True)

MAX_EXAMPLES = 5


# --------------------------------
# General validation functions
# --------------------------------

def is_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def add_error(errors, key, error_type, row_number):
    error_name = f"{key}:{error_type}"

    errors[error_name] = errors.get(error_name, 0) + 1


def check_type(value, expected):
    if expected == "string":
        return isinstance(value, str)

    if expected == "number":
        return is_number(value)

    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected == "list":
        return isinstance(value, list)

    if expected == "dict":
        return isinstance(value, dict)

    return True


# --------------------------------
# Schema definitions
# --------------------------------

REVIEW_SCHEMA = {
    "rating": "number",
    "title": "string",
    "text": "string",
    "asin": "string",
    "parent_asin": "string",
    "user_id": "string",
    "timestamp": "integer",
    "helpful_vote": "integer",
    "verified_purchase": "boolean",
}

META_SCHEMA = {
    "title": "string",
    "description": "list",
    "features": "list",
    "price": "number",
    "images": "list",
    "videos": "list",
    "categories": "list",
    "details": "dict",
    "asin": "string",
    "parent_asin": "string",
}


def expected_type(value, expected):
    if expected == "boolean":
        return isinstance(value, bool)

    return check_type(value, expected)


# --------------------------------
# Record validation
# --------------------------------

def validate_record(row, schema, is_meta, row_number):

    errors = {}

    for key, expected in schema.items():

        if key not in row:
            continue

        value = row[key]

        if value is None:
            continue

        if not expected_type(value, expected):
            add_error(errors, key, "wrong_type", row_number)
            continue

    if is_meta:

        price = row.get("price")

        if is_number(price) and price < 0:
            add_error(errors, "price", "negative", row_number)

        rating = row.get("average_rating")

        if is_number(rating) and not 0 <= rating <= 5:
            add_error(errors, "average_rating", "out_of_range", row_number)

        rating_count = row.get("rating_number")

        if is_number(rating_count) and rating_count < 0:
            add_error(errors, "rating_number", "negative", row_number)

    else:

        rating = row.get("rating")

        if is_number(rating) and not 1 <= rating <= 5:
            add_error(errors, "rating", "out_of_range", row_number)

        helpful = row.get("helpful_vote")

        if is_number(helpful) and helpful < 0:
            add_error(errors, "helpful_vote", "negative", row_number)

        timestamp = row.get("timestamp")

        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            # Amazon Reviews timestamps are Unix milliseconds.
            current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            if timestamp < 0:
                add_error(errors, "timestamp", "negative", row_number)

            elif timestamp > current_ms:
                add_error(errors, "timestamp", "future", row_number)

    return errors


# --------------------------------
# File analysis
# --------------------------------

def analyze(file):

    is_meta = file.name.startswith("meta_")

    cache = cache_dir / (file.name + ".json")

    if cache.exists():
        return json.loads(cache.read_text())

    schema = META_SCHEMA if is_meta else REVIEW_SCHEMA

    length = 0
    error_counts = {}
    examples = {}

    with open(file, "rb") as f:

        for line in f:

            if not line.strip():
                continue

            length += 1

            try:
                row = loads(line)

            except Exception:
                error_counts["INVALID_JSON"] = (
                    error_counts.get("INVALID_JSON", 0) + 1
                )
                continue

            if not isinstance(row, dict):
                error_counts["INVALID_RECORD"] = (
                    error_counts.get("INVALID_RECORD", 0) + 1
                )
                continue

            errors = validate_record(
                row, schema, is_meta, length
            )

            for error_name, count in errors.items():

                error_counts[error_name] = (
                    error_counts.get(error_name, 0) + count
                )

                if error_name not in examples:
                    examples[error_name] = []

                if len(examples[error_name]) < MAX_EXAMPLES:

                    examples[error_name].append({
                        "row": length,
                        "value": str(
                            row.get(error_name.split(":")[0])
                        )[:200]
                    })

            if length % 1_000_000 == 0:
                print(
                    f"{file.name}: {length:,} rows",
                    flush=True
                )

    result = {
        "filename": file.name,
        "length": length,
        "errors": error_counts,
        "examples": examples
    }

    cache.write_text(
        json.dumps(result, ensure_ascii=False)
    )

    print(f"{file.name} finished", flush=True)

    return result


# --------------------------------
# Report generation
# --------------------------------

def write_reports(results, prefix):

    all_errors = sorted({
        error
        for result in results
        for error in result["errors"]
    })

    # Summary report

    with open(
        f"{prefix}_quality_report.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "length",
            *all_errors
        ])

        for result in results:

            writer.writerow([
                result["filename"],
                result["length"],
                *[
                    result["errors"].get(error, 0)
                    for error in all_errors
                ]
            ])

    # Examples report

    with open(
        f"{prefix}_quality_examples.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "error_type",
            "row_number",
            "value"
        ])

        for result in results:

            for error, examples in result["examples"].items():

                for example in examples:

                    writer.writerow([
                        result["filename"],
                        error,
                        example["row"],
                        example["value"]
                    ])


# --------------------------------
# Main
# --------------------------------

if __name__ == "__main__":

    files = sorted(parent_dir.glob("*.jsonl"))

    meta_files = [
        f for f in files
        if f.name.startswith("meta_")
    ]

    review_files = [
        f for f in files
        if not f.name.startswith("meta_")
    ]

    with Pool(processes=3) as pool:

        meta_results = pool.map(analyze, meta_files)

        review_results = pool.map(analyze, review_files)

    write_reports(meta_results, "meta")
    write_reports(review_results, "review")

    print("Quality audit completed.")
