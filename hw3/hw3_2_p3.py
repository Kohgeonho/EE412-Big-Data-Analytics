import sys

from time import time
from collections import defaultdict
from math import sqrt, floor
from itertools import combinations

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
start_time = time()

'''
Step0 Initial Settings

A. Create three data structures
  - node_degrees: Node Degrees, total time required is O(m). 
    type: [Dictionary(hash table)]
  - edg_id_pair: Edge Index (Pair), constructed in O(m) and searched in O(1).
    each pair is sorted by degree and id(int)
    type: [Set(hash table)]
  - edg_id_sing: Edge Index (Single), similar to edg_id_pair. 
    type: [Dictionary(hash table)]
B. variables settings
  - n, m: number of nodes, number of edges
  - threshold: sqrt(m)
'''

edg_id_pair = set()
edg_id_sing = defaultdict(set)

for line in lines:
    n1, n2, ts = line.split('\t')
    edg_id_sing[n1].add(n2)
    edg_id_sing[n2].add(n1)

node_degrees = {
    n: len(fs)
    for n, fs in edg_id_sing.items()
}
edg_id_pair = {
    tuple(sorted((n1, n2), key= lambda x: (node_degrees[x], int(x))))
    for n1 in edg_id_sing.keys()
    for n2 in edg_id_sing[n1]
}

n = len(node_degrees)
m = sum(node_degrees.values()) // 2
threshold = floor(sqrt(m))

''' 
Step1 Count heavy-hitter triangles

A. Consider all triplets of heavy-hitter nodes (there are only O(m3/2) of them)
B. Use edge index to check if all three edges exist in O(1) time
'''

hh_nodes = [n for n, d in node_degrees.items() if d > threshold]
hh_triangles = 0
for c in combinations(hh_nodes, 3):
    n1, n2 ,n3 = sorted(c, key=lambda  x: node_degrees[x])
    hh_triangles += (n1, n2) in edg_id_pair and \
                    (n1, n3) in edg_id_pair and \
                    (n2, n3) in edg_id_pair

'''
Step 2: Count other triangles

A. Consider each edge (v1, v2) where at least one node v is not a heavy hitter
    a. WLOG, assume v1 is not a heavy hitter and v1 ≺ v2
B. Find nodes u1, u2, …,uk  adjacent to v1  (here k < sqrt(m)) 
C. For each ui, check if (ui, v2) exists using edge pair index
D. Count triangle {v1, v2, ui} iff the edge (ui, v2) exists and v1 ≺ v2 ≺ ui 
   (avoids double counting triangles)
'''

other_triangles = 0
for v1, v2 in edg_id_pair:
    if v1 in hh_nodes and v2 in hh_nodes:
        pass
    other_triangles += sum(
        (v2, ui) in edg_id_pair
        for ui in edg_id_sing[v1]
    )

print(hh_triangles + other_triangles)
# print(f"elapsed time: {time() - start_time:.2f}s")
