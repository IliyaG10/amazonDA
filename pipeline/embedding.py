import os
import numpy as np, torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib
import shutil
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import zipfile
import gdown  # pip install gdown

class Embedding:

    def __init__(self, out_dir, con, seed):
        self.con = con
        self.out_dir = out_dir
        self.seed = seed

    def embed_items(self, limit=None, batch_size=256, chunk=20000):
        """a important warning this model will use local all-MiniLM-l6-v2 so you must download the model and make a folder named models and put the model in it"""

        src = f"'{self.out_dir}/item_features_full.parquet'"
        n = self.con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        if limit:
            n = min(n, limit)
        print(f"items to embed: {n:,}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)
        model = SentenceTransformer(
            "models/all-MiniLM-L6-v2",
            device=device,
            local_files_only=True
        )
        model.max_seq_length = 128
        if device == "cuda":
            model.half()  # fp16: faster, less VRAM

        os.makedirs(f"{self.out_dir}/emb", exist_ok=True)
        emb = np.lib.format.open_memmap(
            f"{self.out_dir}/emb/item_emb.npy", mode="w+",
            dtype=np.float16, shape=(n, 384))

        res = self.con.execute(f"""
            SELECT parent_asin, item_text FROM {src}
            ORDER BY parent_asin {f'LIMIT {limit}' if limit else ''}
        """)
        ids, i = [], 0
        while True:
            rows = res.fetchmany(chunk)
            if not rows:
                break
            ids.extend(r[0] for r in rows)
            vec = model.encode([r[1] for r in rows], batch_size=batch_size,
                               normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            emb[i:i + len(rows)] = vec.astype(np.float16)
            i += len(rows)
            print(f"embedded {i:,}/{n:,}")
        emb.flush()
        np.save(f"{self.out_dir}/emb/item_ids.npy", np.array(ids))

    def pca_plot(self, sample=50000, top_cats=8):

        emb = np.load(f"{self.out_dir}/emb/item_emb.npy", mmap_mode="r")
        ids = np.load(f"{self.out_dir}/emb/item_ids.npy", allow_pickle=True)
        rng = np.random.default_rng(self.seed)
        idx = np.sort(rng.choice(len(ids), size=min(sample, len(ids)), replace=False))
        X = np.asarray(emb[idx], dtype=np.float32)

        cat = self.con.execute(f"""
            SELECT parent_asin, main_category
            FROM '{self.out_dir}/item_features_full.parquet'
        """).df().set_index("parent_asin")["main_category"]
        labels = cat.reindex(ids[idx]).fillna("Unknown").values
        top = pd.Series(labels).value_counts().head(top_cats).index
        labels = np.where(np.isin(labels, top), labels, "Other")

        os.makedirs("reports", exist_ok=True)
        os.makedirs("reports/emdeded_items", exist_ok=True)
        # (a) how many components are needed
        full = PCA(n_components=100, random_state=self.seed).fit(X)
        cum = full.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, 101), cum)
        plt.xlabel("components");
        plt.ylabel("cumulative explained variance")
        plt.grid(alpha=.3);
        plt.tight_layout()
        plt.savefig("reports/pca_variance.png", dpi=150);
        plt.close()
        for k in (2, 10, 50, 100):
            print(f"{k} components explain {cum[k - 1]:.1%} of variance")

        # (b) 2D scatter colored by category
        Z = PCA(n_components=2, random_state=self.seed).fit_transform(X)
        plt.figure(figsize=(9, 7))
        for c in sorted(set(labels)):
            m = labels == c
            plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=.4, label=c)
        plt.legend(markerscale=6, fontsize=8)
        plt.xlabel("PC1");
        plt.ylabel("PC2");
        plt.tight_layout()
        plt.savefig("reports/emdeded_items/pca_2d.png", dpi=150);
        plt.close()

    @staticmethod
    def get_model_path():
        """Return models/all-MiniLM-L6-v2. Downloads from Google Drive if needed,
        and accepts any folder layout inside the zip."""
        MODELS_DIR = "models"
        MODEL_NAME = "all-MiniLM-L6-v2"
        MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}"
        MODEL_DRIVE_ID = "1UPFv9gAAGbmPXQnRE3Fo0WZi_8xB0y2W"
        TMP = f"{MODELS_DIR}/_model_tmp"

        def find_root(base):
            for root, _, files in os.walk(base):
                if "config.json" in files and "modules.json" in files:
                    return root
            return None

        if os.path.isfile(f"{MODEL_PATH}/config.json"):
            return MODEL_PATH
        os.makedirs(MODELS_DIR, exist_ok=True)

        root = find_root(MODELS_DIR)  # maybe already unpacked by an earlier run
        if root is None:
            zip_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
            print("Model not found locally, downloading from Google Drive ...")
            gdown.download(id=MODEL_DRIVE_ID, output=zip_path, quiet=False)
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError(
                    "Download failed (Drive may be limiting this file). Download the zip by hand from "
                    f"https://drive.google.com/file/d/{MODEL_DRIVE_ID}/view and unzip it into '{MODELS_DIR}/'")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(MODELS_DIR)
            os.remove(zip_path)
            root = find_root(MODELS_DIR)
            if root is None:
                raise FileNotFoundError("The zip does not contain a sentence-transformers model "
                                        "(config.json and modules.json were not found).")

        # move whatever layout the zip had into models/all-MiniLM-L6-v2
        if os.path.abspath(root) == os.path.abspath(MODELS_DIR):  # files were unpacked directly into models/
            os.makedirs(TMP, exist_ok=True)
            for name in os.listdir(MODELS_DIR):
                if name != "_model_tmp":
                    shutil.move(f"{MODELS_DIR}/{name}", f"{TMP}/{name}")
        else:
            shutil.move(root, TMP)
        shutil.rmtree(MODEL_PATH, ignore_errors=True)
        shutil.move(TMP, MODEL_PATH)
        print("donwlaod has finished")
        return MODEL_PATH

    def embed(self):
        self.get_model_path()
        self.embed_items()
        self.pca_plot()
