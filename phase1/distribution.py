
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path("/media/ghaedi/Data/amazon")
OUTPUT_DIR = Path("EDA_Results")

CHUNK_SIZE = 10_000
SAMPLE_SIZE = 30_000
TOP_CATEGORIES = 15
MAX_NUMERIC_PLOTS = 8
MAX_CATEGORICAL_PLOTS = 8

RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_empty(value):

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, dict)):
        return len(value) == 0

    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))


def normalize_object(value):

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return value


def save_csv(data, path):

    pd.DataFrame(data).to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ==========================================
# 3. READ FILE + EXACT AUDIT + RESERVOIR SAMPLE
# ==========================================

def process_file(file_path):

    print(f"\nProcessing: {file_path.name}")

    file_output = OUTPUT_DIR / file_path.stem
    file_output.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    missing_counts = {}
    column_counts = {}

    # Each row gets a random priority.
    # Keeping the smallest priorities creates
    # a uniform reservoir sample.

    reservoir = pd.DataFrame()
    priorities = np.array([])

    for chunk in pd.read_json(
        file_path,
        lines=True,
        chunksize=CHUNK_SIZE
    ):

        chunk_size = len(chunk)
        total_rows += chunk_size

        # ----------------------------------
        # Missing values: exact full dataset
        # ----------------------------------

        for col in chunk.columns:

            if col not in missing_counts:
                missing_counts[col] = 0
                column_counts[col] = 0

            empty_mask = chunk[col].map(is_empty)

            missing_counts[col] += empty_mask.sum()
            column_counts[col] += chunk_size

        # ----------------------------------
        # Reservoir sampling
        # ----------------------------------

        new_priorities = rng.random(chunk_size)

        chunk = chunk.copy()

        combined = pd.concat(
            [reservoir, chunk],
            ignore_index=True
        )

        combined_priorities = np.concatenate(
            [priorities, new_priorities]
        )

        if len(combined) > SAMPLE_SIZE:

            selected = np.argpartition(
                combined_priorities,
                SAMPLE_SIZE - 1
            )[:SAMPLE_SIZE]

            combined = combined.iloc[selected]
            combined_priorities = combined_priorities[selected]

        reservoir = combined.reset_index(drop=True)
        priorities = combined_priorities

    print(f"Total rows: {total_rows:,}")
    print(f"Sample rows: {len(reservoir):,}")

    if total_rows == 0:
        print("Empty file. Skipping.")
        return

    # ======================================
    # 4. MISSING VALUE REPORT
    # ======================================

    missing_report = []

    for col in missing_counts:

        missing = missing_counts[col]
        total = column_counts[col]

        missing_report.append({

            "column": col,
            "total_rows": total,
            "missing_count": missing,
            "missing_percentage": (
                missing / total * 100 if total else 0
            )

        })

    missing_df = pd.DataFrame(missing_report)

    missing_df.to_csv(
        file_output / "missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 5. DETECT VARIABLE TYPES
    # ======================================

    df = reservoir.copy()

    numeric_columns = []
    categorical_columns = []
    text_columns = []

    for col in df.columns:

        series = df[col].map(normalize_object)

        if pd.api.types.is_bool_dtype(series):
            categorical_columns.append(col)
            df[col] = series.astype("string")
            continue

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid_original = series.notna().sum()
        valid_numeric = converted.notna().sum()

        # At least 90% of non-empty values
        # must be numeric to classify as numeric.

        if (
            valid_original > 0
            and valid_numeric / valid_original >= 0.90
        ):

            numeric_columns.append(col)
            df[col] = converted

        else:

            df[col] = series.astype("string")

            unique_count = series.nunique(dropna=True)

            if unique_count <= 50:
                categorical_columns.append(col)
            else:
                text_columns.append(col)

    print("Numeric:", numeric_columns)
    print("Categorical:", categorical_columns)
    print("Text:", text_columns)

    # ======================================
    # 6. UNIVARIATE NUMERIC ANALYSIS
    # ======================================

    numeric_report = []

    for col in numeric_columns:

        series = df[col].dropna()

        if len(series) == 0:
            continue

        numeric_report.append({

            "column": col,
            "count": len(series),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "variance": series.var(),
            "min": series.min(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
            "max": series.max(),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "iqr": (
                series.quantile(0.75)
                - series.quantile(0.25)
            )

        })

    numeric_df = pd.DataFrame(numeric_report)

    numeric_df.to_csv(
        file_output / "numeric_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 7. UNIVARIATE CATEGORICAL ANALYSIS
    # ======================================

    categorical_report = []

    for col in categorical_columns:

        series = df[col].dropna().astype(str)

        if len(series) == 0:
            continue

        frequencies = series.value_counts()

        categorical_report.append({

            "column": col,
            "unique_values": series.nunique(),
            "mode": frequencies.index[0],
            "mode_frequency": frequencies.iloc[0],
            "mode_percentage": (
                frequencies.iloc[0] / len(series) * 100
            )

        })

        frequencies.head(TOP_CATEGORIES).rename_axis(
            "category"
        ).reset_index(
            name="frequency"
        ).to_csv(
            file_output / f"frequency_{safe_name(col)}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    pd.DataFrame(categorical_report).to_csv(
        file_output / "categorical_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 8. UNIVARIATE TEXT ANALYSIS
    # ======================================

    text_report = []

    for col in text_columns:

        series = df[col].dropna().astype(str)

        if len(series) == 0:
            continue

        lengths = series.str.len()

        text_report.append({

            "column": col,
            "count": len(series),
            "mean_length": lengths.mean(),
            "median_length": lengths.median(),
            "std_length": lengths.std(),
            "min_length": lengths.min(),
            "max_length": lengths.max()

        })

    pd.DataFrame(text_report).to_csv(
        file_output / "text_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 9. NUMERIC DISTRIBUTION PLOTS
    # ======================================

    plot_dir = file_output / "plots"
    plot_dir.mkdir(exist_ok=True)

    for col in numeric_columns[:MAX_NUMERIC_PLOTS]:

        series = df[col].dropna()

        if len(series) == 0:
            continue

        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 4)
        )

        axes[0].hist(
            series,
            bins=40,
            edgecolor="black"
        )

        axes[0].set_title(f"Distribution: {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(series, vert=True)

        axes[1].set_title(f"Boxplot: {col}")
        axes[1].set_ylabel(col)

        fig.tight_layout()

        fig.savefig(
            plot_dir / f"numeric_{safe_name(col)}.png",
            dpi=150
        )

        plt.close(fig)

    # ======================================
    # 10. CATEGORICAL DISTRIBUTION PLOTS
    # ======================================

    for col in categorical_columns[:MAX_CATEGORICAL_PLOTS]:

        frequencies = (
            df[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(TOP_CATEGORIES)
        )

        if len(frequencies) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        frequencies.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("Frequency")

        fig.tight_layout()

        fig.savefig(
            plot_dir / f"category_{safe_name(col)}.png",
            dpi=150
        )

        plt.close(fig)

    # ======================================
    # 11. TEXT LENGTH DISTRIBUTION
    # ======================================

    for col in text_columns[:MAX_CATEGORICAL_PLOTS]:

        lengths = df[col].dropna().astype(str).str.len()

        if len(lengths) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(lengths, bins=40, edgecolor="black")

        ax.set_title(f"Text length distribution: {col}")
        ax.set_xlabel("Number of characters")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        fig.savefig(
            plot_dir / f"text_length_{safe_name(col)}.png",
            dpi=150
        )

        plt.close(fig)

    # ======================================
    # 12. MULTIVARIATE: CORRELATION
    # ======================================

    usable_numeric = [
        col for col in numeric_columns
        if df[col].nunique(dropna=True) > 1
    ]

    if len(usable_numeric) >= 2:

        corr = df[usable_numeric].corr(
            method="spearman"
        )

        corr.to_csv(
            file_output / "spearman_correlation.csv",
            encoding="utf-8-sig"
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        image = ax.imshow(
            corr,
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))

        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax.set_yticklabels(corr.columns)

        ax.set_title("Spearman Correlation Matrix")

        fig.colorbar(image, ax=ax)

        fig.tight_layout()

        fig.savefig(
            plot_dir / "correlation_matrix.png",
            dpi=150
        )

        plt.close(fig)

    # ======================================
    # 13. MULTIVARIATE: SCATTER PLOTS
    # ======================================

    scatter_columns = usable_numeric[:4]

    for i in range(len(scatter_columns)):

        for j in range(i + 1, len(scatter_columns)):

            x = scatter_columns[i]
            y = scatter_columns[j]

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                df[x],
                df[y],
                alpha=0.3,
                s=10
            )

            ax.set_xlabel(x)
            ax.set_ylabel(y)

            ax.set_title(f"{x} vs {y}")

            fig.tight_layout()

            fig.savefig(
                plot_dir / (
                    f"scatter_{safe_name(x)}_"
                    f"{safe_name(y)}.png"
                ),
                dpi=150
            )

            plt.close(fig)

    # ======================================
    # 14. MULTIVARIATE: NUMERIC VS CATEGORY
    # ======================================

    for cat_col in categorical_columns[:4]:

        categories = (
            df[cat_col]
            .value_counts()
            .head(8)
            .index
        )

        for num_col in usable_numeric[:4]:

            plot_data = df[
                [cat_col, num_col]
            ].dropna().copy()

            plot_data[cat_col] = (
                plot_data[cat_col].astype(str)
            )

            plot_data = plot_data[
                plot_data[cat_col].isin(
                    categories.astype(str)
                )
            ]

            if plot_data.empty:
                continue

            groups = [
                plot_data.loc[
                    plot_data[cat_col] == category,
                    num_col
                ].values
                for category in categories.astype(str)
            ]

            groups = [
                group for group in groups if len(group) > 0
            ]

            labels = [
                category for category in categories.astype(str)
                if len(
                    plot_data.loc[
                        plot_data[cat_col] == category,
                        num_col
                    ]
                ) > 0
            ]

            if len(groups) < 2:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.boxplot(
                groups,
                tick_labels=labels
            )

            ax.set_title(f"{num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)

            plt.setp(
                ax.get_xticklabels(),
                rotation=30,
                ha="right"
            )

            fig.tight_layout()

            fig.savefig(
                plot_dir / (
                    f"box_{safe_name(num_col)}_"
                    f"by_{safe_name(cat_col)}.png"
                ),
                dpi=150
            )

            plt.close(fig)

    # ======================================
    # 15. MULTIVARIATE: CATEGORY VS CATEGORY
    # ======================================

    for i in range(min(len(categorical_columns), 4)):

        for j in range(i + 1, min(len(categorical_columns), 4)):

            col1 = categorical_columns[i]
            col2 = categorical_columns[j]

            cross = pd.crosstab(
                df[col1],
                df[col2],
                normalize="index"
            )

            cross.to_csv(
                file_output / (
                    f"crosstab_{safe_name(col1)}_"
                    f"{safe_name(col2)}.csv"
                ),
                encoding="utf-8-sig"
            )

    # ======================================
    # 16. FINISH
    # ======================================

    print(f"Completed: {file_path.name}")
    print(f"Results saved in: {file_output}")


# ==========================================
# 17. PROCESS ALL JSONL FILES
# ==========================================

files = sorted(DATA_DIR.glob("*.jsonl"))

print(f"Found {len(files)} JSONL files.")

for file_path in files:

    try:
        process_file(file_path)

    except Exception as error:

        print(
            f"ERROR in {file_path.name}: {error}"
        )

print("\nEDA completed.")
