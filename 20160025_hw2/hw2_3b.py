import sys
import numpy as np
from time import time

from sklearn.metrics import normalized_mutual_info_score

### Initial Settings
TARGET_USER_ID = 600
TARGET_MOVIE_RANGE = 1000
NUM_SIMILAR = 10
TOP_PREDICTED = 5
SMALL_VALUE = 1e-5

start_time = time()

def cos_sim(A, B):
    normAB = np.linalg.norm(A) * np.linalg.norm(B)
    return (A * B).sum() / normAB if normAB > 0 else 0.

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

ratings = np.array([
    line.split(",") for line in lines
]).astype(float)

## Create Utility Matrix
num_users, num_movies, _, _ = ratings.max(axis=0).astype(int)
utility_matrix = np.zeros((num_users+1, num_movies+1)).astype(np.float)

for u, m, r, t in ratings:
    utility_matrix[int(u), int(m)] = r

## Normalization
normalized_matrix = np.apply_along_axis(
    lambda row: row - (row > 0) * row[row > 0].mean() if sum(row) > 0 else row,
    axis=1,
    arr=utility_matrix
)

### User-based Recommendation
user_similarity = np.apply_along_axis(
    lambda row: cos_sim(row, normalized_matrix[TARGET_USER_ID]),
    axis=1,
    arr=normalized_matrix,
)
similar_user_idx = np.argsort(user_similarity)[-2::-1]
similar_matrix = utility_matrix[similar_user_idx, :TARGET_MOVIE_RANGE+1]
user_based_prediction = np.apply_along_axis(
    lambda col: col[col > 0][:NUM_SIMILAR].mean() \
                if sum(col > 0) >= NUM_SIMILAR else 0.,
    axis=0,
    arr=similar_matrix
)

for idx in np.argsort(user_based_prediction)[-TOP_PREDICTED:][::-1]:
    print(f"{idx}\t{user_based_prediction[idx]}")

end_user_based = time()
print(f"elasped time for user_based: {end_user_based - start_time}s")

### Item-based Recommendation
## Create cosine similarity matrix
movie_norm = np.linalg.norm(normalized_matrix, axis=0)
dot_products = normalized_matrix[:, :TARGET_MOVIE_RANGE+1].T @ normalized_matrix
movie_similarity = dot_products / \
    (movie_norm[None, :] + SMALL_VALUE) / \
    (movie_norm[:TARGET_MOVIE_RANGE+1, None] + SMALL_VALUE)

## Calculate 
def average_movie_rating(row):
    sim_idx = np.argsort(row)[::-1]
    sim_idx = sim_idx[sim_idx > TARGET_MOVIE_RANGE]
    sim_movie_rating = utility_matrix[TARGET_USER_ID, sim_idx]

    rating = sim_movie_rating[sim_movie_rating.nonzero()[0]][:NUM_SIMILAR].mean()
    return rating

item_based_prediction = np.apply_along_axis(
    average_movie_rating,
    axis=1,
    arr=movie_similarity,
)

## filter out non-existing movies
movie_nonexist = utility_matrix[:, :TARGET_MOVIE_RANGE+1].sum(axis=0) > 0
item_based_prediction = item_based_prediction * movie_nonexist

for idx in np.argsort(item_based_prediction)[-TOP_PREDICTED:][::-1]:
    print(f"{idx}\t{item_based_prediction[idx]}")

end_item_based = time()
print(f"elasped time for item_based: {end_item_based - end_user_based}s")
print(f"total elapsed time: {end_item_based - start_time}s")