import sys
import numpy as np

from time import time
from tqdm import tqdm

### Values

m = 500
BATCH_SIZE = 2

### Functions

def create_mask(nonzeros_, shape):
    mask = np.zeros(shape)
    mask[nonzeros_] = 1.
    return mask

def RMSE(M, P, nonzeros):
    return np.linalg.norm(M[nonzeros] - P[nonzeros])

### Initial Settings

start_time = time()
cp = time()

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

num_data = len(lines)
ratings = np.array([
    line.split(",") for line in lines
]).astype(float)

## Create Utility Matrix
num_users, num_movies, _, _ = ratings.max(axis=0).astype(int)
utility_matrix = np.zeros((num_users+1, num_movies+1)).astype(np.float)

for user, mov, r, t in ratings:
    utility_matrix[int(user), int(mov)] = r

nonzeros = utility_matrix.nonzero()
random_idx = np.random.permutation(num_data)
splitting_point = int(num_data * 0.8)
train_set = (
    nonzeros[0][random_idx[:splitting_point]], 
    nonzeros[1][random_idx[:splitting_point]]
)
validation_set = (
    nonzeros[0][random_idx[splitting_point:]], 
    nonzeros[1][random_idx[splitting_point:]]
)
# train_set = np.array(nonzeros)

print(f"create utility matrix: {time() - cp}s")
cp = time()

### Preprocessing
## Normalization
user_mean_matrix = np.apply_along_axis(
    lambda row: row[row > 0].mean() if sum(row) > 0  else 0.,
    axis=1,
    arr=utility_matrix,
)
movie_mean_matrix = np.apply_along_axis(
    lambda col: col[col > 0].mean() if sum(col) > 0 else 0.,
    axis=0,
    arr=utility_matrix,
)
mean_matrix = (user_mean_matrix[:, None] + movie_mean_matrix[None, :]) / 2
normalized_matrix = (utility_matrix - mean_matrix)
M = normalized_matrix * create_mask(train_set, utility_matrix.shape)
validation_matrix = normalized_matrix * create_mask(validation_set, utility_matrix.shape)

print(f"Normalization: {time() - cp}s")
cp = time()

## Perturbation
u, v = num_users+1, num_movies+1
U = np.random.normal(0, 1, (u, m))
V = np.random.normal(0, 1, (m, v))

P = U@V
print(f"    Training RMSE: {RMSE(M, P, train_set)}")
print(f"    Validation RMSE: {RMSE(validation_matrix, P, validation_set)}")

print(f"Perturbation: {time() - cp}s")
cp = time()

### Optimization
## Permutation
U_idx = np.random.permutation(np.array(U.nonzero()).T).T
V_idx = np.random.permutation(np.array(V.nonzero()).T).T

print(f"Permutation: {time() - cp}s")
cp = time()

uv_rate = v // u + 1
for i in tqdm(range(u * m)):
    # Adjust U
    r, s = U_idx[:, i]
    x1 = (V[s, :] ** 2).sum() 
    x2 = (V[s, :] * M[s, :]).sum()
    x3 = (U[r, :] * (V @ V[s, :])).sum()
    U[r, s] += (x2 - x3) / x1   
    # Adjust V
    for j in range(uv_rate):
        r, s = V_idx[:, i * uv_rate + j]
        y1 = (U[:, r] ** 2).sum()
        y2 = (U[:, r] * M[:, s]).sum()
        y3 = (V[:, s] * (U.T @ U[:, r])).sum()
        V[r, s] += (y2 - y3) / y1

    if i % 50 == 0:
        P = U@V
        print(f"    Training RMSE: {RMSE(M, P, train_set)}")
        print(f"    Validation RMSE: {RMSE(validation_matrix, P, validation_set)}")

print(f"Optimization: {time() - cp}s")
cp = time()

## Results
with open(sys.argv[2], 'r') as f:
    pred_lines = f.readlines()

prediction = np.array([
    line.split(",")
    for line in pred_lines
])

result_matrix = U @ V + mean_matrix
prediction = np.apply_along_axis(
    lambda row: (row[0], row[1], result_matrix[int(row[0]), int(row[1])], row[3]),
    axis=1,
    arr=prediction,
)

with open("output.txt", "w") as fw:
    for row in prediction:
        n = fw.write(",".join(row))
        n = fw.write("\n")
