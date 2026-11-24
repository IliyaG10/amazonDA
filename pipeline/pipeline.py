from .database.builder import Builder
from . import clustring, embedding


class Pipeline:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.builder = Builder(raw_dir)
        self.con, self.seed, self.out_dir = self.builder.builder_var()
        self.embedding = embedding.Embedding(self.out_dir, self.con, self.seed)
        self.clustring = clustring.Cluster(self.out_dir, self.con, self.seed)

    def run(self, build=True, embed=True, cluster=True, K=7):
        print("run")
        if build:
            print("build")
            self.builder.build()
            print("finish")
        if embed:
            self.embedding.embed()
        if cluster:
            self.clustring.cluster(K)
