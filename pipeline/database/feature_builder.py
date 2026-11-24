class FeatureBuilder:
    def __init__(self, con, meta_dir, out_dir):
        self.con = con
        self.meta_dir = meta_dir
        self.out_dir = out_dir

    def build_features(self):
        self.con.execute(f"""
               COPY (
                   WITH r AS (
                       SELECT parent_asin,
                              COUNT(*) AS n_reviews,
                              COUNT(DISTINCT user_id) AS n_users,
                              AVG(rating) AS rating_mean,
                              STDDEV(rating) AS rating_std,
                              AVG(verified_purchase::INT) AS verified_ratio,
                              AVG(helpful_vote) AS helpful_mean,
                              AVG(text_len) AS review_len_mean,
                              (MAX(timestamp) - MIN(timestamp)) / 86400000.0
                                  AS active_days
                       FROM '{self.out_dir}/reviews_clean.parquet'
                       WHERE NOT is_test
                       GROUP BY parent_asin
                   )
                   SELECT m.parent_asin, m.main_category, m.item_text,
                          m.has_price, m.has_desc,
                          LN(1 + m.price_filled) AS log_price,
                          m.n_cats,
                          LN(1 + r.n_reviews) AS log_reviews,
                          LN(1 + r.n_users) AS log_users,
                          r.rating_mean,
                          COALESCE(r.rating_std, 0) AS rating_std,
                          r.verified_ratio,
                          r.helpful_mean,
                          r.review_len_mean,
                          r.active_days
                   FROM '{self.out_dir}/meta_clean.parquet' m
                   JOIN r USING (parent_asin)
                   WHERE r.n_reviews >= 5
               )
               TO '{self.out_dir}/item_features.parquet'
               (FORMAT parquet, COMPRESSION zstd)
           """)

    def build_sessions(self, window_days=14, max_group=20, use_verified=True):
        window_ms = window_days * 24 * 3600 * 1000  # Amazon 2023 timestamps are in ms
        ver = "AND r.verified_purchase" if use_verified else ""
        self.con.execute(f"""
            COPY (
                WITH base AS (
                    SELECT r.user_id, r.parent_asin, r.timestamp
                    FROM '{self.out_dir}/reviews_clean.parquet' r
                    JOIN '{self.out_dir}/item_features.parquet' f
                      ON r.parent_asin = f.parent_asin
                    WHERE NOT r.is_test {ver}
                ),
                gaps AS (
                    SELECT *,
                           CASE WHEN timestamp - LAG(timestamp) OVER w > {window_ms}
                                THEN 1 ELSE 0 END AS new_sess
                    FROM base
                    WINDOW w AS (PARTITION BY user_id ORDER BY timestamp)
                ),
                sess AS (
                    SELECT *,
                           SUM(new_sess) OVER (
                               PARTITION BY user_id ORDER BY timestamp
                               ROWS UNBOUNDED PRECEDING) AS session_id
                    FROM gaps
                ),
                sized AS (
                    SELECT *,
                           COUNT(*) OVER (PARTITION BY user_id, session_id) AS size
                    FROM sess
                )
                SELECT user_id, session_id, parent_asin, size
                FROM sized
                WHERE size BETWEEN 2 AND {max_group}
            )
            TO '{self.out_dir}/sessions.parquet'
            (FORMAT parquet, COMPRESSION zstd)
        """)
        print(self.con.execute(f"""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT (user_id, session_id)) AS sessions,
                   COUNT(DISTINCT user_id) AS users,
                   ROUND(AVG(size), 2) AS avg_size
            FROM '{self.out_dir}/sessions.parquet'
        """).df())

    def build_pairs(self, min_users=3):
        sess = f"'{self.out_dir}/sessions.parquet'"
        self.con.execute(f"""
            COPY (
                WITH n AS (
                    SELECT COUNT(*)::DOUBLE AS N
                    FROM (SELECT DISTINCT user_id, session_id FROM {sess})
                ),
                ic AS (
                    SELECT parent_asin, COUNT(*)::DOUBLE AS c
                    FROM {sess}
                    GROUP BY parent_asin
                ),
                pc AS (
                    SELECT a.parent_asin AS item_a, b.parent_asin AS item_b,
                           COUNT(*)::DOUBLE AS n_users,
                           SUM(1.0 / a.size) AS weight
                    FROM {sess} a
                    JOIN {sess} b
                      ON a.user_id = b.user_id
                     AND a.session_id = b.session_id
                     AND a.parent_asin < b.parent_asin
                    GROUP BY a.parent_asin, b.parent_asin
                    HAVING COUNT(*) >= {min_users}
                ),
                mm AS (
                    SELECT DISTINCT ON (parent_asin) parent_asin, main_category
                    FROM '{self.out_dir}/meta_clean.parquet'
                )
                SELECT pc.item_a, pc.item_b, pc.n_users, pc.weight,
                       LN(pc.n_users * n.N / (ica.c * icb.c)) AS pmi,
                       ma.main_category AS cat_a,
                       mb.main_category AS cat_b,
                       (ma.main_category <> mb.main_category) AS cross_cat
                FROM pc
                CROSS JOIN n
                JOIN ic ica ON pc.item_a = ica.parent_asin
                JOIN ic icb ON pc.item_b = icb.parent_asin
                JOIN mm ma ON pc.item_a = ma.parent_asin
                JOIN mm mb ON pc.item_b = mb.parent_asin
                WHERE LN(pc.n_users * n.N / (ica.c * icb.c)) > 0
            )
            TO '{self.out_dir}/pairs.parquet'
            (FORMAT parquet, COMPRESSION zstd)
        """)
        print(self.con.execute(f"""
            SELECT COUNT(*) AS pairs,
                   SUM(cross_cat::INT) AS cross_category_pairs,
                   ROUND(AVG(pmi), 2) AS avg_pmi,
                   MAX(n_users) AS max_users
            FROM '{self.out_dir}/pairs.parquet'
        """).df())

    def add_pair_features(self):
        self.con.execute(f"""
               COPY (
                   WITH long AS (
                       SELECT item_a AS parent_asin, cross_cat
                       FROM '{self.out_dir}/pairs.parquet'
                       UNION ALL
                       SELECT item_b AS parent_asin, cross_cat
                       FROM '{self.out_dir}/pairs.parquet'
                   ),
                   pf AS (
                       SELECT parent_asin,
                              COUNT(*) AS n_pairs,
                              AVG(cross_cat::INT) AS cross_cat_share
                       FROM long
                       GROUP BY parent_asin
                   )
                   SELECT f.*,
                          COALESCE(pf.n_pairs, 0) AS n_pairs,
                          COALESCE(pf.cross_cat_share, 0) AS cross_cat_share
                   FROM '{self.out_dir}/item_features.parquet' f
                   LEFT JOIN pf USING (parent_asin)
               )
               TO '{self.out_dir}/item_features_full.parquet'
               (FORMAT parquet, COMPRESSION zstd)
           """)

    def inspect_pairs(self, k=30):
        base = f"""
            FROM '{self.out_dir}/pairs.parquet' p
            JOIN (SELECT DISTINCT ON (parent_asin) parent_asin, title
                  FROM '{self.out_dir}/meta_clean.parquet') a
              ON p.item_a = a.parent_asin
            JOIN (SELECT DISTINCT ON (parent_asin) parent_asin, title
                  FROM '{self.out_dir}/meta_clean.parquet') b
              ON p.item_b = b.parent_asin
        """
        cols = """p.n_users, ROUND(p.pmi, 2) AS pmi, p.cross_cat,
                  LEFT(a.title, 50) AS title_a, LEFT(b.title, 50) AS title_b"""

        print("TOP by users:")
        print(self.con.execute(
            f"SELECT {cols} {base} ORDER BY p.n_users DESC LIMIT {k}").df())
        print("RANDOM:")
        print(self.con.execute(
            f"SELECT {cols} {base} ORDER BY RANDOM() LIMIT {k}").df())
        print("RANDOM CROSS-CATEGORY:")
        print(self.con.execute(
            f"SELECT {cols} {base} WHERE p.cross_cat ORDER BY RANDOM() LIMIT {k}").df())

    # ---------- Step 8e: diagnose pairs by category ----------
    def diagnose_pairs(self):
        print(self.con.execute(f"""
            SELECT cat_a, cat_b, COUNT(*) AS pairs, ROUND(AVG(pmi), 2) AS avg_pmi
            FROM '{self.out_dir}/pairs.parquet'
            GROUP BY cat_a, cat_b ORDER BY pairs DESC LIMIT 20
        """).df())
        print(self.con.execute(f"""
            WITH long AS (
                SELECT item_a AS item FROM '{self.out_dir}/pairs.parquet'
                UNION ALL SELECT item_b FROM '{self.out_dir}/pairs.parquet')
            SELECT COUNT(*) AS items_with_pairs,
                   MAX(c) AS max_partners,
                   MEDIAN(c) AS median_partners
            FROM (SELECT item, COUNT(*) AS c FROM long GROUP BY item)
        """).df())

    # ---------- Step 8f: remove hubs (mutual top-K) ----------
    def filter_pairs(self, top_k=50, min_users=3):
        self.con.execute(f"""
            COPY (
                WITH p AS (
                    SELECT *, pmi * LN(1 + n_users) AS score
                    FROM '{self.out_dir}/pairs.parquet'
                    WHERE n_users >= {min_users}
                ),
                long AS (
                    SELECT item_a AS item, item_b AS partner, score FROM p
                    UNION ALL
                    SELECT item_b, item_a, score FROM p
                ),
                rk AS (
                    SELECT item, partner,
                           ROW_NUMBER() OVER (PARTITION BY item ORDER BY score DESC) AS r
                    FROM long
                )
                SELECT p.*
                FROM p
                JOIN rk ra ON ra.item = p.item_a AND ra.partner = p.item_b
                JOIN rk rb ON rb.item = p.item_b AND rb.partner = p.item_a
                WHERE ra.r <= {top_k} AND rb.r <= {top_k}
            )
            TO '{self.out_dir}/pairs_filtered.parquet'
            (FORMAT parquet, COMPRESSION zstd)
        """)
        print(self.con.execute(f"""
            SELECT COUNT(*) AS pairs, SUM(cross_cat::INT) AS cross_category_pairs,
                   ROUND(AVG(pmi), 2) AS avg_pmi
            FROM '{self.out_dir}/pairs_filtered.parquet'
        """).df())