import numpy as np 

RANK = 2
M = np.array([ 
    [1,2,3],
    [3,4,5],
    [5,4,3],
    [0,2,4],
    [1,3,5],
])

S1 = M @ M.T
S2 = M.T @ M

print("(a) M@M.T and M.T@M")
print("- M@M.T", S1, sep='\n')
print("- M.T@M", S2, sep='\n', end='\n\n')

w1, v1 = np.linalg.eig(S1)
w2, v2 = np.linalg.eig(S2)

print("(b) eigenpairs for matrices of part (a)")
print("- w1", w1, "- v1", v1, "- w2", w2, "- v2", v2, sep='\n', end='\n\n')

U = v1[:, np.argsort(w1)][:, -RANK:]
V = v2[:, np.argsort(w2)][:, -RANK:]
S = np.sqrt(np.diag(sorted(w1)[-RANK:]))

print("(c) SVD for the matrix M")
print("- U", U, "- V", V, "- S", S, sep="\n", end='\n\n')

sv2 = S[0,0]
S[0,0] = 0
approximate = U @ S @ V.T

print("(d) approximation to the matrix M")
print(approximate, end='\n\n')

sv1 = S[-1,-1]
retained_energy = sv1**2 / (sv1**2 + sv2**2)

print("(e) retained energy")
print(retained_energy)