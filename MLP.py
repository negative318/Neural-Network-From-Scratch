
from activateFuncion import *


import numpy as np
import matplotlib.pyplot as plt
import keras


class NeuralNetwork:
  def __init__(self, layers_size, activations,loss, l_rate):
    self.layers_size = layers_size
    self.activations = activations
    self.loss = loss
    self.l_rate = l_rate
    self.num_layer = len(layers_size)
    self.W = [0]
    self.b = [0]
    for i in range(1,self.num_layer,1):
      self.W.append(np.random.randn(layers_size[i-1],layers_size[i]))
      self.b.append(np.random.randn(layers_size[i],1))


  def forward(self, inputs):
    self.A = [inputs]
    self.Z = [0]
    for i in range(1,self.num_layer,1):
      z = np.dot(self.W[i].T, self.A[-1]) + self.b[i]
      self.Z.append(z)
      self.A.append(self.activeFuncion(z,self.activations[i]))
    return self.A[-1]

  def backpropagation(self,output):
    E = []
    dW = []
    db = []
    for i in range(self.num_layer - 1, 0, -1):
      if i == self.num_layer - 1:
        E.append(self.derivative(output,self.Z[i], self.activations[i]))
        dW.insert(0, np.dot(self.A[i-1], E[-1].T))
        db.insert(0, np.sum(E[-1], axis=1, keepdims=True))
      else:
        E.append(self.derivative(np.dot(self.W[i+1], E[-1]),self.Z[i],self.activations[i]))
        dW.insert(0, np.dot(self.A[i-1], E[-1].T))
        db.insert(0, np.sum(E[-1], axis=1, keepdims=True))
    dW.insert(0,0)
    db.insert(0,0)
    E.append(np.dot(self.W[1], E[-1]))
    for i in range(self.num_layer):
        self.W[i] -= dW[i] * self.l_rate
        self.b[i] -= db[i] * self.l_rate
    return E[-1]


  def activeFuncion(self,Z,active):
    if active == "sigmoid":
      return sigmoid.forward(Z)
    elif active == "relu":
      return ReLU.forward(Z)
    elif active == "tanh":
      return tanh.forward(Z)
    elif active == "softmax":
      return softmax.forward(Z)


  def derivative(self,E,Z,active):
    if active == "sigmoid":
      return sigmoid.backpropagation(E)
    elif active == "relu":
      return E * ReLU.backpropagation(Z)
    elif active == "tanh":
      return tanh.backpropagation(E)
    elif active == "softmax":
      return 1/E.size * (self.A[-1] - E)


  def cost(self,ouput,func):
      if func == 'MAE':
          return np.abs(self.A[-1] - ouput)
      elif func == 'MSE':
          return (self.A[-1] - ouput)**2/2
      elif func == 'crossEntropy':
          return -np.sum(ouput * np.log(self.A[-1]+1e-6))/ouput.shape[1]
      elif func == 'binaryCrossEntropy':
          return -np.sum(ouput * np.log(self.A[-1]+1e-6) + (1 - ouput) * np.log(1 - self.A[-1]+1e-6))


  def get_accuracy(self,predictions,output):
    return np.sum(predictions == output) / output.size


  def train(self,input,output,val_in, val_out, epochs,batch_size =128):
    for i in range(epochs):
      for j in range(0,input.shape[1],batch_size):
        input_batch = input[:,j:j+batch_size]
        output_batch = output[:,j:j+batch_size]
        self.forward(input_batch)
        self.backpropagation(output_batch)
        if i % 100 == 0 and j == 0:
          print(i,"loss: ", self.cost(output_batch,self.loss),
                "Accuracy train: ", self.get_accuracy(np.argmax(self.A[-1],0),np.argmax(output_batch,0)),
                "Accuracy validation: ", self.test(val_in,val_out)
              )


  def test(self,val_in,val_out):
    self.forward(val_in)
    return (self.get_accuracy(np.argmax(self.A[-1],0),np.argmax(val_out,0)))
