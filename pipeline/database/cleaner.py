class Cleaner:
    def __init__(self, con, meta_dir,rev_dir, out_dir):
        self.con = con
        self.meta_dir = meta_dir
        self.rev_dir = rev_dir
        self.out_dir = out_dir


    def clean_reviews(self):
        self.con.execute(f"""
            COPY (
                SELECT DISTINCT ON (user_id, parent_asin)
                       user_id, parent_asin, timestamp, rating,
                       verified_purchase, helpful_vote, text,
                       LENGTH(text) AS text_len,
                       (hash(user_id) % 10 = 0) AS is_test
                FROM '{self.rev_dir}/*.parquet'
                WHERE user_id IS NOT NULL
                  AND parent_asin IS NOT NULL
                  AND text IS NOT NULL
                  AND TRIM(text) <> ''
                ORDER BY user_id, parent_asin, timestamp
            )
            TO '{self.out_dir}/reviews_clean.parquet'
            (FORMAT parquet, COMPRESSION zstd)
        """)


    def clean_meta(self):
        self.con.execute(f"""
            COPY (
                WITH m AS (
                    SELECT parent_asin, title, average_rating,
                           rating_number, price,
                           COALESCE(main_category, categories[1], 'Unknown')
                               AS main_category,
                           ARRAY_TO_STRING(features, ' ') AS feat_txt,
                           ARRAY_TO_STRING(description, ' ') AS desc_txt,
                           ARRAY_LENGTH(categories) AS n_cats,
                           images
                    FROM '{self.meta_dir}/*.parquet'
                    WHERE title IS NOT NULL
                      AND TRIM(title) <> ''
                ),
                med AS (
                    SELECT main_category, MEDIAN(price) AS med_price
                    FROM m
                    GROUP BY main_category
                )
                SELECT m.*,
                       (price IS NOT NULL) AS has_price,
                       (desc_txt IS NOT NULL AND desc_txt <> '') AS has_desc,
                       COALESCE(price, med.med_price) AS price_filled,
                       TRIM(title || ' ' || COALESCE(feat_txt, '') || ' '
                            || LEFT(COALESCE(desc_txt, ''), 800))
                            AS item_text
                FROM m
                LEFT JOIN med USING (main_category)
            )
            TO '{self.out_dir}/meta_clean.parquet'
            (FORMAT parquet, COMPRESSION zstd)
        """)