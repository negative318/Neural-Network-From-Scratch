import numpy as np

class ReLU:
    def __init__(self):
        pass
    def forward(self,input):
        return np.maximum(0, input)
    def backpropagation(self, input):
        return (input > 0).astype(float)

class sigmoid:
    def __init__(self):
        pass
    def forward(self,input):
        return np.where(input >= 0, 1 / (1 + np.exp(-input)), np.exp(input) / (1 + np.exp(input)))
    def backpropagation(self, output):
        s = self.forward(output)
        return s * (1 - s)

class tanh:
    def __init__(self):
        pass
    def forward(self,input):
        return np.tanh(input)
    def backpropagation(self,output):
        return 1 - np.tanh(output) ** 2
    

class softmax:
    def __init__(self):
        pass
    def forward(input):
        input_max = np.max(input, axis=0, keepdims=True)
        exp_input = np.exp(input - input_max)
        A = exp_input / np.sum(exp_input, axis=0, keepdims=True)
        return A