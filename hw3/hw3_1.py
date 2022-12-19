import sys
from time import time
from pyspark import SparkConf, SparkContext

NUM_NODES = 1000
BETA = 0.9

### Initial settings
sc = SparkContext(conf=SparkConf())
lines = sc.textFile(sys.argv[1])
start_time = time()

edges = lines.map(lambda r: r.split())
matrix_row = edges.groupByKey().mapValues(list)
matrix = matrix_row.flatMap(
    lambda r: [(r[0], e, 1 / len(r[1])) for e in r[1]]
)
v = [1 / NUM_NODES] * NUM_NODES

def update_PR(v):

    def matmul(r):
        col, row, m_value = r
        return int(row), m_value * v[int(col)-1]

    matrix_mul = matrix.map(matmul)
    matrix_mul = matrix_mul.reduceByKey(lambda a, b: a + b)
    return matrix_mul.map(
        lambda r: (r[0], BETA * r[1] + (1 - BETA) * 1 / NUM_NODES)
    )

for _ in range(50):
    for i, w in update_PR(v).collect():
        v[i - 1] = w

answer = [(i+1, score) for i, score in enumerate(v)]

for page, score in sorted(answer, key=lambda x:x[1], reverse=True)[:10]:
    print(f"{page}\t{score:.5f}")
# print(f"elapsed time: {time() - start_time:.2f}s")

sc.stop()
