import json
from multiprocessing import Pool
from pathlib import Path
from typing import TypedDict

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads


parent_dir = Path("/media/ghaedi/Data/amazon")
output_file = Path("data_report.json")


class FileReport(TypedDict):
    filename: str
    length: int
    present: dict[str, int]
    empty: dict[str, int]


def is_empty(value: object) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, dict)):
        return len(value) == 0

    return False


def analyze(file: Path) -> FileReport:
    length = 0
    present: dict[str, int] = {}
    empty: dict[str, int] = {}

    print(f"[START] {file.name}", flush=True)

    with file.open("rb") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"{file.name}, line {line_number}: invalid JSON"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"{file.name}, line {line_number}: "
                    "expected a JSON object"
                )

            length += 1

            # کلیدهای هر ردیف بررسی می‌شوند، نه فقط ردیف اول.
            for key, value in row.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{file.name}, line {line_number}: "
                        "expected a string key"
                    )

                present[key] = present.get(key, 0) + 1

                if is_empty(value):
                    empty[key] = empty.get(key, 0) + 1

            # فقط نمایش پیشرفت؛ پردازش متوقف نمی‌شود.
            if length % 1_000_000 == 0:
                print(
                    f"[PROGRESS] {file.name}: "
                    f"{length:,} rows processed; still running",
                    flush=True,
                )

    # فقط بعد از رسیدن به انتهای فایل چاپ می‌شود.
    print(
        f"[FINISHED] {file.name}: {length:,} total rows",
        flush=True,
    )

    return {
        "filename": file.name,
        "length": length,
        "present": present,
        "empty": empty,
    }


def main() -> None:
    files = sorted(parent_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found in: {parent_dir}"
        )

    report: dict[str, dict[str, FileReport]] = {
        "review": {},
        "meta": {},
    }

    with Pool(processes=3) as pool:
        for result in pool.imap_unordered(
            analyze, files, chunksize=1
        ):
            filename = result["filename"]
            group = "meta" if filename.startswith("meta_") else "review"

            report[group][filename] = result

    for group in report:
        report[group] = dict(sorted(report[group].items()))

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to: {output_file.resolve()}", flush=True)


if __name__ == "__main__":
    main()
