import numpy as np

STOP_THRESHOLD = 1e-5

def incremental_analysis(M, v, e, beta):
    new_v = np.zeros(v.shape) / v.shape[0]
    iterations = 0
    while np.linalg.norm(v - new_v) > STOP_THRESHOLD:
        v = new_v
        new_v = beta * M @ new_v + (1 - beta) * e
        iterations += 1

    elements = ['a','b','c','d']
    print(f"total iterations : {iterations}")
    print(f"Page Rank :", end=' ')
    for i, ele in enumerate(v.T[0]):
        print(f"{elements[i]}= {ele:.4f}", end=', ')
    print()

## Exercise 5.1.2

print("\nExercise 5.1.2\n")
M = np.array([
    [1/3, 1/2, 0  ],
    [1/3, 0  , 1/2],
    [1/3, 1/2, 1/2],
])
beta = 0.8
v = np.ones((3,1)) / 3
e = np.ones((3,1)) / 3
incremental_analysis(M, v, e, beta)

## Exercise 5.3.1

print("\nExercise 5.3.1")
M = np.array([
    [0  , 1/2, 1  ,0  ],
    [1/3, 0  , 0  ,1/2],
    [1/3, 0  , 0  ,1/2],
    [1/3, 1/2, 0  ,0  ],
])
beta = 0.8

# (a)
print("\n(a)")
v = np.ones((4,1)) / 4
e = np.array([1,0,0,0]).reshape((4,1))
incremental_analysis(M, v, e, beta)

# (b)
print("\n(b)")
v = np.ones((4,1)) / 4
e = np.array([1,0,1,0]).reshape((4,1)) / 2
incremental_analysis(M, v, e, beta)

