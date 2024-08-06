from autograd import *
from activateFuncion import *


import numpy as np
import matplotlib.pyplot as plt
import keras


class NeuralNetwork:
  def __init__(self, layers_size, activations,lossFunction, l_rate):
    self.layers_size = layers_size
    self.activations = activations
    self.lossFunction = lossFunction
    self.l_rate = l_rate
    self.num_layer = len(layers_size) - 1
    self.W = [Tensor(np.random.randn(layers_size[i], layers_size[i + 1]), requires_grad=True) for i in range(self.num_layer)]
    self.b = [Tensor(np.random.randn(1,layers_size[i + 1]), requires_grad=True) for i in range(self.num_layer)]



  def forward(self, inputs):
    self.A = [inputs]
    for i in range(self.num_layer):
      Z = self.A[-1].dot(self.W[i]) + self.b[i]
      self.A.append(self.activeFuncion(Z,self.activations[i]))
    return self.A[-1]

  def backpropagation(self,output):
    self.A[-1].backward(output)
    self.update_parameter()

  def update_parameter(self):
     for i in range(self.num_layer):
        self.W[i].data -= self.W[i].grad * self.l_rate
        self.W[i].grad = np.zeros_like(self.W[i].data)
        self.b[i].data -= self.b[i].grad * self.l_rate
        self.b[i].grad = np.zeros_like(self.b[i].data)

  def activeFuncion(self,Z,active):
    if active == "sigmoid":
      return Z.sigmoid()
    elif active == "relu":
      return Z.relu()
    elif active == "tanh":
      return Z.tanh()
    elif active == "softmax":
      return Z.softmax()


  def cost(self,output,func):
      if func == 'MAE':
          return np.abs(self.A[-1] - output)
      elif func == 'MSE':
          return (self.A[-1] - output)**2/2
      elif func == 'crossEntropy':
        return -np.sum(output * np.log(self.A[-1].data + 1e-6)) / output.shape[0]
      elif func == 'binaryCrossEntropy':
          return -np.sum(output * np.log(self.A[-1]+1e-6) + (1 - output) * np.log(1 - self.A[-1]+1e-6))


  def get_accuracy(self,predictions,output):
    return np.sum(predictions == output) / output.size


  def train(self,x_train,y_train,x_val, y_val, epochs,batch_size =128):
    for i in range(epochs):
      for j in range(0,x_train.shape[0],batch_size):
        x_train_batch = Tensor(x_train[j:j+batch_size,:], requires_grad=True)
        
        y_train_batch = y_train[j:j+batch_size,:]
        self.forward(x_train_batch)
        self.backpropagation(y_train_batch)

      if i % 100 == 0:
        (loss_train, accuracy_train) = self.test(x_train,y_train)
        (loss_val, accuracy_val) = self.test(x_val,y_val)
        print("epochs:", i, "loss train:", loss_train, "Accuracy train:", accuracy_train,
              "loss val:", loss_val, "Accuracy validation: ", accuracy_val)


  def test(self, val_in, val_out):
      val_in = Tensor(val_in, requires_grad= True)
      self.forward(val_in)
      loss = np.mean(self.cost(val_out, self.lossFunction))
      return (loss, self.get_accuracy(np.argmax(self.A[-1].data.T, 0), np.argmax(val_out.T, 0)))