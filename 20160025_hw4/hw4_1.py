import sys
import numpy as np
# from time import time
# from tqdm import tqdm

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim) 

        self.Sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.dSigmoid = lambda y: y * (1 - y)
        
    def Forward(self, Input):
        '''Implement forward propagation'''

        Hidden = self.Sigmoid(Input @ self.W1)
        Output = self.Sigmoid(Hidden @ self.W2)
        self.Hidden = Hidden

        return Output
    
    def Backward(self, Input, Output, Label):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''

        dLoss = Output - Label
        dW2 = self.Hidden.T @ (self.dSigmoid(Output) * dLoss)
        dHidden = (self.dSigmoid(Output) * dLoss) @ self.W2.T
        dW1 = Input.T @ (self.dSigmoid(self.Hidden) * dHidden)

        self.W2 -= dW2 * self.learning_rate
        self.W1 -= dW1 * self.learning_rate
    
    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Output, Label)        

'''Hyperparameters'''
iteration = 10000
learning_rate = 1e-3

'''Get Dataset'''
train_dataset = np.loadtxt(sys.argv[1], delimiter=',')
test_dataset = np.loadtxt(sys.argv[2], delimiter=',')

train_data, train_cls = train_dataset[:, :-1], train_dataset[:, -1]
test_data, test_cls = test_dataset[:, :-1], test_dataset[:, -1]

train_label = np.zeros((1000, 10))
test_label = np.zeros((1000, 10))

train_label[np.arange(1000), train_cls.astype(int)] = 1
test_label[np.arange(1000), test_cls.astype(int)] = 1

'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

'''Train the network for the number of iterations'''
'''Implement function to measure the accuracy'''
def accuracy(data, cls):
    Output = Network.Forward(data)
    Pred = Output.argmax(axis=1)
    return sum(Pred == cls) / len(cls)

train_acc = []
train_loss = []
# pbar = tqdm(range(iteration))
# for i in pbar:
for i in range(iteration):
    Network.Train(train_data, train_label)
    # if i % 10 == 0:
    #     Output = Network.Forward(train_data)
    #     Loss = ((Output - train_label) ** 2).sum() / 2000
    #     train_acc.append(accuracy(train_data, train_cls))
    #     train_loss.append(Loss)
        # pbar.set_description(f"Test Acc: {accuracy(test_data, test_cls):.4f}")

print(accuracy(train_data, train_cls))
print(accuracy(test_data, test_cls))
print(iteration)
print(learning_rate)

# import matplotlib.pyplot as plt

# x = np.arange(iteration // 10)
# y = train_acc

# plt.plot(x, y)
# plt.show()