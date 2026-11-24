from .convert import Converter
from .cleaner import Cleaner
from .feature_builder import FeatureBuilder

class Builder:

    def __init__(self, raw_dir, seed=42,
                 memory_limit="8GB", threads=4):
        self.converter = Converter(raw_dir, seed, memory_limit, threads)
        self.seed = seed
        self.con ,self.out_dir, self.rev_dir, self.meta_dir = self.converter.protector_loc()
        self.cleaner = Cleaner(self.con, self.meta_dir, self.rev_dir, self.out_dir)
        self.feature_builder = FeatureBuilder(self.con, self.meta_dir, self.out_dir)

    def report(self):
        n = self.con.execute(f"""
            SELECT COUNT(*)
            FROM '{self.out_dir}/item_features.parquet'
        """).fetchone()[0]

        print(f"items in final feature table: {n:,}")

        print(self.con.execute(f"""
            SELECT main_category,
                   COUNT(*) AS items,
                   ROUND(AVG(has_price::INT), 2) AS share_with_price
            FROM '{self.out_dir}/item_features.parquet'
            GROUP BY main_category
            ORDER BY items DESC
        """).df())


    def build(self):
        self.converter.make_file()
        self.converter.parquet_converter()
        self.cleaner.clean_reviews()
        self.cleaner.clean_meta()
        self.feature_builder.build_features()
        self.report()
        self.feature_builder.build_sessions(window_days=14, max_group=20)
        self.feature_builder.build_pairs(min_users=3)
        self.feature_builder.add_pair_features()
        self.feature_builder.inspect_pairs()
        self.feature_builder.diagnose_pairs()
        self.feature_builder.filter_pairs()

    def builder_var(self):
        return self.con, self.seed, self.out_dir