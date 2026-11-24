
from pipeline.pipeline import Pipeline

if __name__ == "__main__":
    pipe = Pipeline("/media/ghaedi/Data/amazon")
    pipe.run(build=False, embed=False, cluster=False, K=7)