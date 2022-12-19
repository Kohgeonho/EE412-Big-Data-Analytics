import sys
import numpy as np
from time import time

NUM_NODES = 1000

start_time = time()
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

M = np.zeros((NUM_NODES, NUM_NODES))
for line in lines:
    row, col = line.split()
    M[int(col)-1, int(row)-1] = 1
M = M / M.sum(axis=0)[None, :]

beta = 0.9
v = np.ones((NUM_NODES,1)) / NUM_NODES
e = np.ones((NUM_NODES,1)) / NUM_NODES

for _ in range(50):
    v = beta * M @ v + (1 - beta) * e

sorted_v = sorted(
    [(i, s) for i, s in enumerate(v[:, 0])],
    key= lambda x: x[1],
    reverse=True,
)
for i, s in sorted_v[:10]:
    print(f"{i+1}\t{s}")
print(f"elapsed time: {time() - start_time}s")