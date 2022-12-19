import sys
import re
import time
import numpy as np
from math import sqrt, floor

THRESHOLD = 0.9
NUM_BANDS = 6
ROW_SIZE = 20
NUM_SIGNS = NUM_BANDS * ROW_SIZE
start_time = time.time()

def is_prime(p):
    """
    Decide if it is prime or not
    """
    for i in range(2, floor(sqrt(p))):
        if p%i == 0:
            return False
    return True

def JaccardSimilarity(s1, s2):
    """
    Calculate the Jaccard Similarity between two list of items.

    ex.
        s1: ['abc', 'def', 'abc', 'efg']
        s2: ['abc', 'efg', 'gab']
        return: 0.5
    """
    set1 = set(s1)
    set2 = set(s2)
    common = len(set1.intersection(set2))
    set1.update(set2)
    return common / len(set1)

## Read articles / Preprocessing
with open("articles.txt", "r") as f:
    lines = f.readlines()

#  1. transform alphabet into lower-case
#  2. remove non-alphabetic characters
#  3. transform multi-space into single-space
articles = [(
        l.split()[0], 
        re.sub(
            ' +', 
            ' ', 
            re.sub(
                r'[^a-z\s]', 
                '', 
                l[l.find(" ")+1:-2].lower()
            )
        )
    ) 
    for l in lines
]
#  Transform text data into shingles
shingles = [(
        key,
        [txt[i:i+3] for i in range(len(txt)-2)]
    )
    for key, txt in articles
]
#  Set of shingles from all documents
shingle_set = set(
    s
    for t, sh_list in shingles
    for s in sh_list
)
shingle_dic = {
    s:i 
    for i, s in enumerate(shingle_set)
}
n = len(shingle_set)
num_doc = len(articles)

## Create SDMatrix (Signature-Document Matrix)
SDMatrix = np.zeros((num_doc, n))
for i in range(num_doc):
    for s in shingles[i][1]:
        SDMatrix[i, shingle_dic[s]] = 1.

## Create hash functions
#  select NUM_SIGNS(120) distinct a's and b's.
c = n
while not is_prime(c):
    c += 1
A = np.random.choice(c, NUM_SIGNS)
B = np.random.choice(c, NUM_SIGNS)

## Create MHMatrix(Min-Hashed Matrix)
MHMatrix = np.apply_along_axis(
    lambda col: (
        (A * col.nonzero()[0][:, np.newaxis] + B) % c
    ).min(axis=0),
    axis=1,
    arr=SDMatrix,
)

## Get Candidate Pairs from MHMatrix
#  use python dictionary to hash band into buckets
Candidates = []
for i in range(NUM_BANDS):
    Bucket = {}
    for j, band in enumerate(MHMatrix[:, i*ROW_SIZE:(i+1)*ROW_SIZE]):
        if not str(band) in Bucket:
            Bucket[str(band)] = []
        Bucket[str(band)].append(j)
    Candidates += [sorted(pair) for b, pair in Bucket.items() if len(pair) > 1]

## Verify those candidate pairs by calculating exact Jaccard Similarity
#  assume there are no triples but only pairs in Candidates
SimDocs = set()
for i, j in Candidates:
    if JaccardSimilarity(shingles[i][1], shingles[j][1]) > THRESHOLD:
        SimDocs.add((i, j))

for i, j in SimDocs:
    print(f"{articles[i][0]}\t{articles[j][0]}")
# print(f"elapsed time: {time.time() - start_time}s")
