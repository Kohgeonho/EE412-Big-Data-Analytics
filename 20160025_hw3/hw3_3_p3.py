import sys
import numpy as np

from time import time

with open(sys.argv[1], 'r') as f:
    features = f.readlines()
with open(sys.argv[2], 'r') as f:
    labels = f.readlines()

features = np.array([f.split(',') for f in features]).astype(np.uint8)
labels = np.array(labels).astype(np.int8)
start_time = time()

class SVM():

    def __init__(self, features, labels, k=10):
        '''
        Step0 train/test split for k-fold cross validation

        A. create features/labels matrix
        B. split features and labels into k separate matrices
        - k: number of folds. 10 in this task.
        - num_data: number of total datas. Assume that num_data is a multiple of k.
        '''
        self.num_data, self.num_features = features.shape
        self.chunk_size = self.num_data // k

        self.k = k
        self.features = features.reshape(k, self.chunk_size, self.num_features)
        self.labels = labels.reshape(k, self.chunk_size)

    def fitpredict(self, test_idx, C=1e-2, gamma=1e-2, early_stopping_stage=10):
        '''
        Step1 SVM

        0. Train/Test split.
        1. Values for the parameters C and η.
        2. Initial values for w, including the (d + 1)st component b.

        Then, we repeatedly:
        (a) Compute the partial derivatives of f(w,b) with respect to the wj’s.
        (b) Adjust the values of w by subtracting η ∂f from each wj. ∂wj
        (c) Stop when training acc doesn't rise for 10 iterations.
        '''

        train_idx = np.ones(self.k).astype(np.bool8)
        train_idx[test_idx] = False
        train_size = self.chunk_size * (self.k - 1)
        
        train_X = self.features[train_idx].reshape(train_size, self.num_features)
        train_X = np.hstack([train_X, np.ones((train_size, 1))])
        train_y = self.labels[train_idx].reshape(-1)

        self.test_X = self.features[test_idx]
        self.test_X = np.hstack([self.test_X, np.ones((self.chunk_size, 1))])
        self.test_y = self.labels[test_idx]

        w = np.random.normal(0, 1, self.num_features+1)
        max_acc, max_acc_idx = 0, 0
        train_acc, iters = 0, 0

        while train_acc <= max_acc and max_acc_idx + early_stopping_stage > iters:
            output = w @ train_X.T * train_y
            train_acc = sum(output > 0) / train_size
            error = sum(1 - output[output < 1])

            df = w - C * train_y @ (train_X * (output < 1)[:, None])
            w -= gamma * df

            if train_acc > max_acc:
                max_acc = train_acc
                max_acc_idx = iters
            iters += 1

        self.w = w

        pred = w @ self.test_X.T * self.test_y
        test_acc = sum(pred > 0) / self.chunk_size

        return test_acc

    def run(self, C, gamma):
        '''
        Step2 K-fold

        A. repeat fit-predict k times.
        B. Get the average accuracy, C, gamma values from total iterations
        '''

        accuracy = np.array([
            self.fitpredict(i, C, gamma)
            for i in range(self.k)
        ])
        
        print(accuracy.mean())
        print(C)
        print(gamma)

svm = SVM(features, labels)
svm.run(C=0.01, gamma=0.01)
# print(f"elapsed time: {time() - start_time:.2f}s")
