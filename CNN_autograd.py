from activateFuncion import *
from autograd import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
class Convolutional:

  

  def __init__(self, input_shape, kernel_size, output_depth, l_rate, activeFuncion = ""):
    input_depth, input_height, input_width = input_shape
    output_height = (input_height - kernel_size) + 1
    output_width = (input_width - kernel_size) + 1
    self.depth = output_depth

    self.input_shape = input_shape
    self.kernel_shape = (output_depth, input_depth, kernel_size, kernel_size)
    self.bias_shape = (output_depth, output_height, output_width)
    self.output_shape = (output_depth, output_height, output_width)

    self.kernel = Tensor(np.ones((self.kernel_shape)), requires_grad= True)
    self.bias = Tensor(np.ones((self.bias_shape)), requires_grad= True)

    self.l_rate = l_rate
    self.active_text = activeFuncion


  def active(self,input):
      if self.active_text == "":
        return input
      if self.active_text == "relu":
            return input.relu()
      elif self.active_text == "sigmoid":
          return input.sigmoid()
      elif self.active_text == "tanh":
          return input.tanh()
      else:
          raise ValueError(f"Unknown activation function: {self.active_text}")


  def forward(self, input):
    self.input = input
    output = self.input.conv(self.kernel) + self.bias
    return self.active(output)


  def update_parameter(self):
    # print(self.bias.grad)
    self.kernel.data -= self.kernel.grad * self.l_rate
    self.kernel.grad = np.zeros_like(self.kernel.data)
    self.bias.data -= self.bias.grad * self.l_rate
    self.bias.grad = np.zeros_like(self.bias.data)

class MaxPoolingLayer:
  def __init__(self, pool_size):
    self.pool_size = pool_size

  def forward(self, input): # input 4 chiều là 1 tensor
    output = input.maxpooling(self.pool_size)
    return output
  def update_parameter(self):
     pass

#input 4 chiều là 1 tensor
class Flattening():
  def __init__(self):
    pass
  def forward(self, input):
    output = input.flatten()
    return output
  def update_parameter(self):
     pass




class Model:
  def __init__(self, layers):
    self.layers = layers
  def train(self, x_train,y_train,x_val,y_val, batch_size, epochs):
    print("x_train: ", x_train.data.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.data.shape)
    print("y_val: ", y_val.shape)
    
    for e in range(epochs):
      for i in range(0,x_train.shape[0] ,batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size,:]
        output = Tensor(x_batch, requires_grad= True)
        for layers in self.layers:
          output = layers.forward(output)
          


        output.backward(y_batch)

        for layers in self.layers:
          layers.update_parameter()
      (loss_train, accuracy_train) = self.test(x_train, y_train)
      (loss_val, accuracy_val) = self.test(x_val, y_val)
      print("epochs: ", e, "loss_train: ",loss_train, "accuracy_train:", accuracy_train,
            "loss_val", loss_val, "accuracy_validation:", accuracy_val)


  def check(self, x_test, y_test):
      output = Tensor(x_test, requires_grad= True)
      for layers in self.layers:
        output = layers.forward(output)
      
      loss = -np.sum(y_test * np.log(output.data+1e-6)) / y_test.shape[0]
      accuracy = self.get_accuracy(np.argmax(output.data.T,0),np.argmax(y_test.T,0))
      return (np.mean(loss), accuracy)
  
  def get_accuracy(self,predictions,output):
    return np.sum(predictions == output) / output.size


  def test(self, x_test, y_test):
      class_dirs = ['aloevera','banana','bilimbi','cantaloupe','cassava','coconut','corn','cucumber','curcuma','eggplant']
      output = Tensor(x_test, requires_grad= True)
      for layers in self.layers:
        output = layers.forward(output)
      loss = -np.sum(y_test * np.log(output.data+1e-6)) / y_test.shape[0]

      y_pred = np.argmax(output.data.T, 0)
      y_true = np.argmax(y_test.T, 0)
      y_pred_labels = np.array([class_dirs[i] for i in y_pred])
      y_true_labels = np.array([class_dirs[i] for i in y_true])
      
      
      print("(",loss, self.get_accuracy(y_pred, y_true),")")
      cm = confusion_matrix(y_true, y_pred)
      plt.figure(figsize=(10, 8))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_dirs, yticklabels=class_dirs)
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title('Test Set Confusion Matrix')
      plt.show()
