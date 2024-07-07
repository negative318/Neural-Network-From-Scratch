
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = x_train /255.0
x_test = x_test /255.0
x_validation = x_train[:6000]
y_validation = y_train[:6000]
x_train = x_train[6000:]
y_train = y_train[6000:]

x_validation = x_validation.reshape(x_validation.shape[0],-1).T
x_train = x_train.reshape(x_train.shape[0],-1).T
x_test = x_test.reshape(x_test.shape[0],-1).T


def one_hot(Y):
  one_hot_Y = np.zeros((Y.size,10))
  one_hot_Y[np.arange(Y.size),Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

y_train = one_hot(y_train)
y_validation = one_hot(y_validation)
y_test = one_hot(y_test)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(x_validation.shape,y_validation.shape)

class activationFunction:
  def sigmoid(Z):
      return 1 / (1 + np.exp(-Z))

  def relu(Z):
      return np.maximum(0, Z)

  def tanh(Z):
      return np.tanh(Z)

  def softmax(Z):
    Z_max = np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A

class derivative:
    def sigmoid(Z):
      s = activationFunction.sigmoid(Z)
      return s * (1 - s)

    def relu(Z):
      return (Z > 0).astype(float)

    def tanh(Z):
      return 1 - np.tanh(Z) ** 2

class NeuralNetwork:
  def __init__(self, layers_size, activations, l_rate):
    self.layers_size = layers_size
    self.activations = activations
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

  def back_prop(self,output):
    E = []
    dW = []
    db = []
    for i in range(self.num_layer - 1, 0, -1):
      if i == self.num_layer - 1:
        E.append(1/output.size * (self.A[-1] - output))
        dW.insert(0, np.dot(self.A[i-1], E[-1].T))
        db.insert(0, np.sum(E[-1], axis=1, keepdims=True))
      else:
        E.append(np.dot(self.W[i+1], E[-1]))
        E[-1][self.Z[i] <= 0] = 0
        dW.insert(0, np.dot(self.A[i-1], E[-1].T))
        db.insert(0, np.sum(E[-1], axis=1, keepdims=True))
    dW.insert(0,0)
    db.insert(0,0)
    for i in range(self.num_layer):
        self.W[i] -= dW[i] * self.l_rate
        self.b[i] -= db[i] * self.l_rate

  def activeFuncion(self,Z,active):
    if active == "sigmoid":
      return activationFunction.sigmoid(Z)
    elif active == "relu":
      return activationFunction.relu(Z)
    elif active == "tanh":
      return activationFunction.tanh(Z)
    elif active == "softmax":
      return activationFunction.softmax(Z)


  def derivative(self,Z,active):
    if active == "sigmoid":
      return derivative.sigmoid(Z)
    elif active == "relu":
      return derivative.relu(Z)
    elif active == "tanh":
      return derivative.tanh(Z)
    elif active == "softmax":
      return 1/Z.size * (self.A[-1] - Z)


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


  def train(self,input,output,val_in, val_out, epochs):
    for i in range(epochs):
      batch = 128
      for j in range(0,input.shape[1],batch):
        input_batch = input[:,j:j+batch]
        output_batch = output[:,j:j+batch]
        self.forward(input_batch)
        self.back_prop(output_batch)
        if i % 10 == 0 and j == 0:
          print(i,"loss: ", self.cost(output_batch,"crossEntropy"),
                "Accuracy train: ", self.get_accuracy(np.argmax(self.A[-1],0),np.argmax(output_batch,0)),
                "Accuracy validation: ", self.test(val_in,val_out)
              )


  def test(self,val_in,val_out):
    self.forward(val_in)
    return (self.get_accuracy(np.argmax(self.A[-1],0),np.argmax(val_out,0)))

nn = NeuralNetwork(layers_size=[28*28, 128, 64, 10],activations = ["","relu", "relu", "softmax"], l_rate=0.01)
nn.train(x_train, y_train,x_validation, y_validation, epochs=1000)

nn.test(x_test,y_test)