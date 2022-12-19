import numpy as np

THRESHOLD = 1e-5

A = np.array([
    [1, 1, 1],
    [1, 2, 3],
    [1, 3, 6],
])

## (a) power iteration to find an approximate value of the principal eigenvector
eig_vec1 = np.array([1, 1, 1]).reshape((3, 1))
diff = 1

while diff > THRESHOLD:
    new_eig_vec = A @ eig_vec1
    new_eig_vec = new_eig_vec / np.linalg.norm(new_eig_vec)
    diff = np.linalg.norm(eig_vec1 - new_eig_vec)
    eig_vec1 = new_eig_vec

print("(a) first eigen vector")
print(eig_vec1, end="\n\n")

eig_val1 = (eig_vec1.T @ A @ eig_vec1)[0,0]

print("(b) first eigen value")
print(eig_val1, end='\n\n')

A2 = A - eig_val1 * (eig_vec1 @ eig_vec1.T)

print("(c) new matrix")
print(A2, end='\n\n')

def get_eig_pair(M):
    eig_vec = np.array([1, 1, 1]).reshape((3, 1))
    diff = 1

    while diff > THRESHOLD:
        new_eig_vec = M @ eig_vec
        new_eig_vec = new_eig_vec / np.linalg.norm(new_eig_vec)
        diff = np.linalg.norm(eig_vec - new_eig_vec)
        eig_vec = new_eig_vec

    eig_val = eig_vec.T @ M @ eig_vec
    new_M = M - eig_val * (eig_vec @ eig_vec.T)

    return eig_vec, eig_val[0,0], new_M

eig_vec2, eig_val2, A3 = get_eig_pair(A2)

print("(d) second eigen pair")
print(eig_vec2)
print(eig_val2, end='\n\n')

eig_vec3, eig_val3, A4 = get_eig_pair(A3)

print("(e) third eigen pair")
print(eig_vec3)
print(eig_val3, end='\n\n')

V = np.concatenate([eig_vec1, eig_vec2, eig_vec3], axis=1)
U = np.diag([eig_val1, eig_val2, eig_val3])

print("Verification: reconstructed matrix")
print(V @ U @ V.T)