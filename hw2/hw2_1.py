import sys
import numpy as np
from time import time
from pyspark import SparkConf, SparkContext

### Initial settings
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
K = int(sys.argv[2])

sc = SparkContext(conf=SparkConf())

documents = np.array([line.split() for line in lines]).astype(np.float)
spark_documents = sc.parallelize(list(range(len(documents))))

### Initialization of clusters
initial_centroids = [0]
while len(initial_centroids) < K:
    new_centroid = np.argmax(
        np.apply_along_axis(
            lambda d: min(
                np.linalg.norm(d - documents[i])
                for i in initial_centroids
            ),
            axis=1,
            arr=documents,
        )
    )
    initial_centroids.append(new_centroid)

### K-means Algorithm
clustered_documents = spark_documents.map(
    lambda d: (
        np.argmin([
            np.linalg.norm(documents[d] - documents[c])
            for c in initial_centroids
        ]),
        d
    )
)
clustroids = clustered_documents.groupByKey().mapValues(list)

### Average Diameter
clustroid_diameter = [
    (
        c, 
        np.linalg.norm(
            documents[docs, None, :] - documents[None, docs, :].astype(np.float),
            axis=2,
        ).astype(np.float).max().astype(np.float)
    )
    for c, docs in clustroids.collect()
]
print(sum(d for c, d in clustroid_diameter) / K)

sc.stop()
