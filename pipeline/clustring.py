import os
import numpy as np, re, torch
from collections import Counter
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
import scipy.sparse as sp
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi
from .embedding import Embedding


class Cluster:

    def __init__(self, out_dir, con, seed):
        self.out_dir = out_dir
        self.con = con
        self.seed = seed

    def make_sample(self, n=150000):
        d = f"{self.out_dir}/emb"
        ids = np.load(f"{d}/item_ids.npy", allow_pickle=True)
        pos = {a: i for i, a in enumerate(ids)}
        pair_items = self.con.execute(f"""
            SELECT item_a AS a FROM '{self.out_dir}/pairs_filtered.parquet'
            UNION SELECT item_b FROM '{self.out_dir}/pairs_filtered.parquet'
        """).df()["a"].tolist()
        chosen = {pos[a] for a in pair_items if a in pos}
        rng = np.random.default_rng(self.seed)
        rest = np.setdiff1d(np.arange(len(ids)), np.fromiter(chosen, dtype=int))
        need = max(0, n - len(chosen))
        chosen |= set(rng.choice(rest, size=min(need, len(rest)), replace=False).tolist())
        idx = np.array(sorted(chosen))
        sid = ids[idx]
        emb = np.load(f"{d}/item_emb.npy", mmap_mode="r")
        np.save(f"{d}/sample_emb.npy", np.asarray(emb[idx], dtype=np.float32))

        self.con.register("sid_tbl", pd.DataFrame({"parent_asin": sid}))
        rc = f"'{self.out_dir}/reviews_clean.parquet'"
        df = self.con.execute(f"""
            WITH age AS (
                SELECT parent_asin,
                       ((SELECT MAX(timestamp) FROM {rc}) - AVG(timestamp)) / 86400000.0
                           AS review_age_days
                FROM {rc}
                WHERE NOT is_test AND parent_asin IN (SELECT parent_asin FROM sid_tbl)
                GROUP BY parent_asin
            )
            SELECT f.*, age.review_age_days
            FROM '{self.out_dir}/item_features_full.parquet' f
            JOIN sid_tbl USING (parent_asin)
            LEFT JOIN age USING (parent_asin)
        """).df()
        df = df.drop_duplicates("parent_asin").set_index("parent_asin").loc[sid].reset_index()
        self.con.register("sample_df", df)
        self.con.execute(f"COPY (SELECT * FROM sample_df) TO '{d}/sample.parquet' (FORMAT parquet)")
        print(f"sample: {len(df):,} items ({len(pair_items):,} items come from pairs)")

    def _read_sample(self):
        return self.con.execute(f"SELECT * FROM '{self.out_dir}/emb/sample.parquet'").df()

    def tfidf_svd(self, n_components=100):
        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        vec = TfidfVectorizer(max_features=50000, min_df=5, max_df=0.5, ngram_range=(1, 2),
                              stop_words="english", sublinear_tf=True, dtype=np.float32)
        T = vec.fit_transform(df["item_text"])
        sp.save_npz(f"{d}/tfidf.npz", T)
        np.save(f"{d}/tfidf_terms.npy", vec.get_feature_names_out())
        svd = TruncatedSVD(n_components, random_state=self.seed)
        np.save(f"{d}/tfidf_svd.npy", svd.fit_transform(T).astype(np.float32))
        print(f"TF-IDF shape {T.shape}, density {T.nnz / (T.shape[0] * T.shape[1]):.4%}")
        print(f"SVD({n_components}) explains {svd.explained_variance_ratio_.sum():.1%} of variance")

    def compare_neighbors(self, n_examples=5, k=5):

        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        E = np.load(f"{d}/sample_emb.npy")
        T = sp.load_npz(f"{d}/tfidf.npz")
        rng = np.random.default_rng(self.seed)
        for i in rng.choice(len(df), n_examples, replace=False):
            en = np.argsort(-(E @ E[i]))[1:k + 1]
            tn = np.argsort(-(T @ T[i].T).toarray().ravel())[1:k + 1]
            print("\nITEM:", df.item_text[i][:80])
            print("  embedding neighbors:", *[f"\n    - {df.item_text[j][:70]}" for j in en])
            print("  tf-idf neighbors:   ", *[f"\n    - {df.item_text[j][:70]}" for j in tn])

    def build_representation(self, n_pca=50, struct_weight=0.5, top_cats=15):

        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        E = np.load(f"{d}/sample_emb.npy")
        pca = PCA(n_pca, random_state=self.seed)
        Ze = pca.fit_transform(E)
        print(f"PCA({n_pca}) keeps {pca.explained_variance_ratio_.sum():.1%} of embedding variance")

        cols = ["log_price", "n_cats", "rating_mean", "rating_std", "review_len_mean",
                "review_age_days", "verified_ratio", "helpful_mean", "has_price", "has_desc"]
        S = df[cols].astype(float)
        S["review_len_mean"] = np.log1p(S["review_len_mean"])
        S["helpful_mean"] = np.log1p(S["helpful_mean"])
        S = S.fillna(S.median())
        S = S.clip(S.quantile(0.01), S.quantile(0.99), axis=1)
        S = StandardScaler().fit_transform(S)

        Ze = Ze / np.sqrt(Ze.var(0).sum())  # each block has total variance 1
        S = S / np.sqrt(S.var(0).sum())
        np.save(f"{d}/emb_block.npy", Ze);
        np.save(f"{d}/struct_block.npy", S)
        np.save(f"{d}/rep.npy", np.hstack([Ze, struct_weight * S]).astype(np.float32))

        top = df["main_category"].value_counts().head(top_cats).index.tolist()
        onehot = np.stack([(df["main_category"] == c).values for c in top], 1).astype(np.int8)
        np.save(f"{d}/cat_onehot.npy", onehot)
        json.dump({"struct_cols": cols, "categories": top}, open(f"{d}/feature_names.json", "w"))

    def select_k(self, ks=(5, 10, 15, 20, 30, 40, 50), sil_n=20000):

        X = np.load(f"{self.out_dir}/emb/rep.npy")
        sub = np.random.default_rng(self.seed).choice(len(X), min(sil_n, len(X)), replace=False)
        rows = []
        for k in ks:
            a = MiniBatchKMeans(k, random_state=self.seed, n_init=3, batch_size=4096).fit(X)
            b = MiniBatchKMeans(k, random_state=self.seed + 1, n_init=3, batch_size=4096).fit(X)
            rows.append(dict(k=k, inertia=a.inertia_,
                             silhouette=silhouette_score(X[sub], a.labels_[sub]),
                             davies_bouldin=davies_bouldin_score(X[sub], a.labels_[sub]),
                             ari_two_seeds=adjusted_rand_score(a.labels_, b.labels_)))
            print(rows[-1])
        r = pd.DataFrame(rows)
        os.makedirs("reports", exist_ok=True)
        r.to_csv("reports/k_selection.csv", index=False)
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
        for a_, c, t in zip(ax, ["inertia", "silhouette", "davies_bouldin", "ari_two_seeds"],
                            ["inertia (elbow)", "silhouette (higher)", "Davies-Bouldin (lower)",
                             "stability ARI (higher)"]):
            a_.plot(r.k, r[c], "o-");
            a_.set_title(t);
            a_.set_xlabel("k");
            a_.grid(alpha=.3)
        plt.tight_layout();
        plt.savefig("reports/k_selection.png", dpi=150);
        plt.close()
        print(r.round(3))

    def stability(self, k, n_runs=5, frac=0.8):
        d = f"{self.out_dir}/emb"
        Ze, S = np.load(f"{d}/emb_block.npy"), np.load(f"{d}/struct_block.npy")
        X = np.load(f"{d}/rep.npy")
        mk = lambda kk, s: MiniBatchKMeans(kk, random_state=s, n_init=5, batch_size=4096)
        ref = mk(k, self.seed).fit(X).labels_
        np.save(f"{d}/labels_k{k}.npy", ref)
        rng = np.random.default_rng(self.seed)

        seeds = [ari(ref, mk(k, self.seed + i).fit(X).labels_) for i in range(1, n_runs)]
        subs = []
        for i in range(n_runs):
            idx = rng.choice(len(X), int(frac * len(X)), replace=False)
            subs.append(ari(ref, mk(k, self.seed + i).fit(X[idx]).predict(X)))
        print(f"different seeds     ARI mean {np.mean(seeds):.3f} (min {np.min(seeds):.3f})")
        print(f"{int(frac * 100)}% subsamples     ARI mean {np.mean(subs):.3f} (min {np.min(subs):.3f})")
        for w in (0.0, 0.25, 1.0):
            lab = mk(k, self.seed).fit(np.hstack([Ze, w * S])).labels_
            print(f"struct weight {w:<4}  ARI vs reference {ari(ref, lab):.3f}")
        for kk in (int(k * 0.5), int(k * 1.5), k * 2):
            lab = mk(kk, self.seed).fit(X).labels_
            print(f"k={kk:<3} vs k={k}   AMI {ami(ref, lab):.3f}  (granularity change)")

    def interpret_clusters(self, k, top_terms=12):
        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        labels = np.load(f"{d}/labels_k{k}.npy")
        df["cluster"] = labels
        df["price"] = np.where(df["has_price"], np.expm1(df["log_price"]), np.nan)
        T = sp.load_npz(f"{d}/tfidf.npz").tocsr()
        terms = np.load(f"{d}/tfidf_terms.npy", allow_pickle=True)
        glob = np.asarray(T.mean(0)).ravel()

        s = df.groupby("cluster").agg(
            size=("parent_asin", "size"), rating=("rating_mean", "mean"),
            price_median=("price", "median"), age_days=("review_age_days", "mean"),
            n_pairs=("n_pairs", "mean"), cross_share=("cross_cat_share", "mean"),
            top_category=("main_category", lambda x: x.value_counts().index[0]),
            top_cat_share=("main_category", lambda x: x.value_counts(normalize=True).iloc[0]))
        s["top_words"] = [", ".join(terms[np.argsort(-(np.asarray(T[labels == c].mean(0)).ravel() - glob))[:top_terms]])
                          for c in s.index]
        os.makedirs("reports", exist_ok=True)
        s.round(3).to_csv("reports/cluster_summary.csv")
        pd.crosstab(df["cluster"], df["main_category"], normalize="index").round(3).to_csv(
            "reports/cluster_vs_category.csv")
        print("NMI(cluster, category) =", round(nmi(df["main_category"], labels), 3))
        print("NMI(cluster, rounded rating) =", round(nmi(df["rating_mean"].round().fillna(0), labels), 3))
        pd.set_option("display.max_columns", None, "display.width", 250, "display.max_colwidth", 90)
        print(s.round(2))

    def analyze_pairs(self, k):

        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        E = np.load(f"{d}/sample_emb.npy")
        labels = np.load(f"{d}/labels_k{k}.npy")
        pos = {a: i for i, a in enumerate(df["parent_asin"])}
        p = self.con.execute(f"""SELECT item_a, item_b, n_users, pmi, cross_cat
                                 FROM '{self.out_dir}/pairs_filtered.parquet'""").df()
        p = p[p.item_a.isin(pos) & p.item_b.isin(pos)].copy()
        ia, ib = p.item_a.map(pos).values, p.item_b.map(pos).values
        p["cos"] = (E[ia] * E[ib]).sum(1)
        p["same_cluster"] = labels[ia] == labels[ib]
        rng = np.random.default_rng(self.seed)
        ra, rb = rng.integers(0, len(df), 100000), rng.integers(0, len(df), 100000)
        chance = ((np.bincount(labels) / len(labels)) ** 2).sum()
        print(f"pairs analysed: {len(p):,}")
        print(f"mean cosine: pairs {p.cos.mean():.3f} vs random items {(E[ra] * E[rb]).sum(1).mean():.3f}")
        print(f"same cluster: pairs {p.same_cluster.mean():.1%} vs chance {chance:.1%}")
        hi, lo = p.cos.quantile(0.75), p.cos.quantile(0.40)
        p["type"] = np.where(p.cos >= hi, "similar", np.where(p.cos <= lo, "complementary", "related"))
        print(p.groupby("type").agg(pairs=("cos", "size"), cos=("cos", "mean"),
                                    same_cluster=("same_cluster", "mean"), cross_cat=("cross_cat", "mean")).round(3))
        for t in ("similar", "complementary"):
            print(f"\nEXAMPLES: {t}")
            for _, r in p[p.type == t].sample(8, random_state=self.seed).iterrows():
                print(f"  {df.item_text[pos[r.item_a]][:55]}  <->  {df.item_text[pos[r.item_b]][:55]}")
        self.con.register("typed_df", p)
        self.con.execute(f"COPY (SELECT * FROM typed_df) TO '{self.out_dir}/pairs_typed.parquet' (FORMAT parquet)")

    def embed_sample(self, batch_size=256, min_repeat=100):

        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        if "item_text_raw" not in df.columns:
            df["item_text_raw"] = df["item_text"]
        split = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+|\s{2,}", t) if s.strip()]
        cnt = Counter(s for t in df["item_text_raw"] for s in set(split(t)))
        boiler = {s for s, c in cnt.items() if c >= min_repeat and len(s) > 25}
        print(f"boilerplate sentences removed: {len(boiler):,}")
        print("examples:", [s[:70] for s in list(boiler)[:5]])

        def clean(t):
            t = t.replace("Product Description", " ")
            return " ".join(s for s in split(t) if s not in boiler)[:1000]

        df["item_text"] = [clean(t) or t for t in df["item_text_raw"]]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(Embedding.get_model_path(), device=dev, local_files_only=True)
        model.max_seq_length = 128
        if dev == "cuda":
            model.half()
        E = model.encode(df["item_text"].tolist(), batch_size=batch_size,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True).astype(np.float32)
        np.save(f"{d}/sample_emb.npy", E)
        self.con.register("clean_df", df)
        self.con.execute(f"COPY (SELECT * FROM clean_df) TO '{d}/sample.parquet' (FORMAT parquet)")
        print("done: sample_emb.npy and sample.parquet (item_text is now cleaned)")

    def cluster_stability(self, k, n_runs=10, frac=0.8):

        d = f"{self.out_dir}/emb"
        X = np.load(f"{d}/rep.npy")
        ref = np.load(f"{d}/labels_k{k}.npy")
        rng = np.random.default_rng(self.seed)
        scores = np.zeros((n_runs, k))
        for r in range(n_runs):
            idx = rng.choice(len(X), int(frac * len(X)), replace=False)
            lab = MiniBatchKMeans(k, random_state=self.seed + r, n_init=5,
                                  batch_size=4096).fit(X[idx]).predict(X)
            new_sizes = np.bincount(lab, minlength=k)
            for c in range(k):
                m = ref == c
                inter = np.bincount(lab[m], minlength=k)
                scores[r, c] = (inter / (m.sum() + new_sizes - inter)).max()
        out = pd.DataFrame({"cluster": range(k), "size": np.bincount(ref, minlength=k),
                            "jaccard_mean": scores.mean(0), "jaccard_min": scores.min(0)})
        out["verdict"] = np.where(out.jaccard_mean > .75, "stable",
                                  np.where(out.jaccard_mean > .6, "moderate", "unstable"))
        os.makedirs("reports", exist_ok=True)
        out.round(3).to_csv("reports/cluster_stability.csv", index=False)
        print(out.round(3))

    def export_pair_validation(self, n=150):
        df = self._read_sample()
        title = dict(zip(df["parent_asin"], df["item_text"].str[:90]))
        p = self.con.execute(f"SELECT * FROM '{self.out_dir}/pairs_typed.parquet'").df()
        s = p.groupby("type", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n // 3), random_state=self.seed))
        out = pd.DataFrame({"type_by_cosine": s["type"], "n_users": s["n_users"],
                            "cos": s["cos"].round(3),
                            "item_a": s["item_a"].map(title), "item_b": s["item_b"].map(title),
                            "your_label": ""})
        out.sample(frac=1, random_state=self.seed).to_csv("reports/pair_validation.csv", index=False)
        print("fill 'your_label' with: C = functional complement, V = variant/series/substitute, "
              "T = same taste (media), N = noise")

    def plot_clusters(self, k, n=20000):

        d = f"{self.out_dir}/emb"
        X, lab = np.load(f"{d}/rep.npy"), np.load(f"{d}/labels_k{k}.npy")
        idx = np.random.default_rng(self.seed).choice(len(X), min(n, len(X)), replace=False)
        Z = PCA(2, random_state=self.seed).fit_transform(X[idx])
        plt.figure(figsize=(9, 7))
        plt.scatter(Z[:, 0], Z[:, 1], c=lab[idx], cmap="tab10", s=2, alpha=.5)
        plt.xlabel("PC1");
        plt.ylabel("PC2");
        plt.tight_layout()
        plt.savefig("reports/clusters_pca.png", dpi=150);
        plt.close()

    def export_projector(self, k, n=20000):
        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        X = np.load(f"{d}/rep.npy")
        lab = np.load(f"{d}/labels_k{k}.npy")
        idx = np.sort(np.random.default_rng(self.seed).choice(len(df), min(n, len(df)), replace=False))
        os.makedirs("reports/projector", exist_ok=True)
        np.savetxt("reports/projector/vectors.tsv", X[idx], fmt="%.5f", delimiter="\t")
        clean = lambda s: s.str.replace(r"[\t\r\n]+", " ", regex=True).str[:80]
        meta = pd.DataFrame({
            "title": clean(df["item_text"].iloc[idx]),
            "cluster": [f"C{c}" for c in lab[idx]],
            "category": df["main_category"].iloc[idx].values,
            "rating": df["rating_mean"].iloc[idx].round(1).astype(str).values})
        meta.to_csv("reports/projector/metadata.tsv", sep="\t", index=False)
        print(f"saved {len(idx):,} points in reports/projector/")

    def plot_3d(self, k, n=20000, method="umap", color="cluster"):

        d = f"{self.out_dir}/emb"
        df = self._read_sample()
        X = np.load(f"{d}/rep.npy")
        lab = np.load(f"{d}/labels_k{k}.npy")
        idx = np.sort(np.random.default_rng(self.seed).choice(len(df), min(n, len(df)), replace=False))
        if method == "umap":
            import umap
            Z = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                          random_state=self.seed).fit_transform(X[idx])
        else:
            from sklearn.decomposition import PCA
            Z = PCA(3, random_state=self.seed).fit_transform(X[idx])
        p = pd.DataFrame({
            "x": Z[:, 0], "y": Z[:, 1], "z": Z[:, 2],
            "cluster": [f"C{c}" for c in lab[idx]],
            "category": df["main_category"].iloc[idx].values,
            "rating": df["rating_mean"].iloc[idx].round(2).values,
            "title": df["item_text"].iloc[idx].str[:70].values}).sort_values("cluster")
        fig = px.scatter_3d(p, x="x", y="y", z="z", color=color, hover_name="title",
                            hover_data=["category", "rating", "cluster"], opacity=0.7)
        fig.update_traces(marker_size=2)
        fig.update_layout(legend_itemsizing="constant", margin=dict(l=0, r=0, t=0, b=0))
        os.makedirs("reports", exist_ok=True)
        fig.write_html(f"reports/clusters_3d_{method}_{color}.html", include_plotlyjs=True)
        print(f"saved reports/clusters_3d_{method}_{color}.html")

    def cluster(self, K=7):

        self.make_sample(n=150000)  # overwrites sample files, so embed_sample comes next
        self.embed_sample()
        self.tfidf_svd()
        self.compare_neighbors()
        self.build_representation()
        self.select_k(ks=range(4, 16))
        self.stability(k=K)  # saves labels_k{K}.npy, must be first
        self.cluster_stability(k=K)
        self.interpret_clusters(k=K)
        self.analyze_pairs(k=K)  # saves pairs_typed.parquet
        self.export_pair_validation()  # needs pairs_typed.parquet
        self.plot_clusters(k=K)
        self.export_projector(k=K)
        self.plot_3d(k=K, method="umap", color="cluster")
        self.plot_3d(k=K, method="umap", color="category")  # same points, colored by category
        self.plot_3d(k=K, method="pca", color="cluster")
