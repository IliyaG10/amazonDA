import os
import duckdb
import glob


class Converter:
    def __init__(self, raw_dir, seed, memory_limit, threads):
        """this will make a folder named parquet with two folder named meta and reviews
            the memory and threads limit set for the computer equipment
            raw_dir is the path of location of jsonl files"""
        self.out_dir = "parquet"
        self.raw_dir, self.out_dir, self.seed = raw_dir, self.out_dir, seed
        self.rev_dir = f"{self.out_dir}/reviews"
        self.meta_dir = f"{self.out_dir}/meta"
        os.makedirs(self.rev_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs("tmp_duckdb", exist_ok=True)
        self.con = duckdb.connect()
        self.con.execute(f"SET memory_limit='{memory_limit}'; SET threads={threads};")
        self.con.execute("SET temp_directory='tmp_duckdb';")
        self.files = {"meta": [], "reviews": []}

    def make_file(self):
        """separate the meta and reviews files"""
        files = glob.glob(f"{self.raw_dir}/*.jsonl")

        self.files["meta"] = [
            f for f in files if os.path.basename(f).startswith("meta")
        ]

        self.files["reviews"] = [
            f for f in files if not os.path.basename(f).startswith("meta")
        ]

    def _name(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def parquet_converter(self):
        """path attribute is a string for the absolute path of the location of Amazon jsonl file """
        # convert the
        print("converting review files start")
        for f in self.files["reviews"]:
            out = f"{self.rev_dir}/{self._name(f)}.parquet"
            if os.path.exists(out):
                continue  # restartable
            self.con.execute(f"""
                      COPY (SELECT user_id, parent_asin, asin, timestamp, rating,
                                   verified_purchase, helpful_vote, title, text
                            FROM read_json('{f}', format='newline_delimited',
                              columns={{user_id:'VARCHAR', parent_asin:'VARCHAR',
                                asin:'VARCHAR', timestamp:'BIGINT', rating:'DOUBLE',
                                verified_purchase:'BOOLEAN', helpful_vote:'INTEGER',
                                title:'VARCHAR', text:'VARCHAR'}}, ignore_errors=true))
                      TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)
                    """)
            print("reviews done:", out)

        print("converting meta files start")

        for f in self.files["meta"]:
            out = f"{self.meta_dir}/{self._name(f)}.parquet"
            if os.path.exists(out):
                continue
            self.con.execute(f"""
                      COPY (SELECT parent_asin, main_category, title, store,
                                   average_rating, rating_number,
                                   TRY_CAST(price AS DOUBLE) AS price,
                                   features, description, categories, images
                            FROM read_json('{f}', format='newline_delimited',
                              columns={{parent_asin:'VARCHAR', main_category:'VARCHAR',
                                title:'VARCHAR', store:'VARCHAR', average_rating:'DOUBLE',
                                rating_number:'INTEGER', price:'VARCHAR',
                                features:'VARCHAR[]', description:'VARCHAR[]',
                                categories:'VARCHAR[]', images:'JSON'}},
                              ignore_errors=true))
                      TO '{out}' (FORMAT parquet, COMPRESSION zstd)
                    """)
            print("meta done:", out)

    def protector_loc(self):
        return self.con, self.out_dir, self.rev_dir, self.meta_dir
