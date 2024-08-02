from activateFuncion import *
from autograd import *
import numpy as np

class Convolutional:

  
  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def __init__(self, input_shape, kernel_size, output_depth, l_rate, activeFuncion = ""):
    input_depth, input_height, input_width = input_shape
    output_height = (input_height - kernel_size) + 1
    output_width = (input_width - kernel_size) + 1
    self.depth = output_depth


    self.input_shape = input_shape
    self.kernel_shape = (output_depth, input_depth, kernel_size, kernel_size)
    self.bias_shape = (output_depth, output_height, output_width)
    self.output_shape = (output_depth, output_height, output_width)

    self.kernel = Tensor(np.random.randn(*self.kernel_shape), requires_grad= True)
    self.bias = Tensor(np.random.randn(*self.bias_shape), requires_grad= True)

    self.l_rate = l_rate
    self.active_text = activeFuncion



      
    
  # img Tensor(Hi, Wi)
  # kernel Tensor(Hk,Wk)
  def conv(self, image, kernel, padding):
    output = image.conv(kernel,padding = padding)
    return output


  
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


  # input = (num,in_chanel,height, width)
  # kernel = Tensor(out_chanel,in_chanel, height, width)
  # bias = Tensor(chanel,height,width)
  # output = Tensor(num, out_chanel, height, width)
  def forward(self, input):
    self.input = input
    output = self.input.conv(self.kernel)
    return output


  def update_parameter(self):
      self.kernel.data -= self.kernel.grad * self.l_rate
      self.kernel.data = np.zeros_like(self.kernel.data)
      self.bias.data -= self.bias.grad * self.l_rate
      self.bias.grad = np.zeros_like(self.bias.data)

class MaxPoolingLayer:
  def __init__(self, pool_size):
    self.pool_size = pool_size

  def forward(self, input): # input 4 chiều là 1 tensor
    output = input.maxpooling(self.pool_size)
    return output

#input 4 chiều là 1 tensor
class Flattening():
  def __init__(self):
    pass
  def forward(self, input):
    output = input.flatten()
    return output.T


class Model:
  def __init__(self, CNN_layers, nn_layer):
    self.CNN_layers = CNN_layers
    self.nn_layer = nn_layer
    self.flatten = Flattening()
  def train(self, x_train,y_train,x_val,y_val, batch_size, epochs):
    self.batch_size = batch_size
    self.epochs = epochs
    print("x_train: ", x_train.data.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.data.shape)
    print("y_val: ", y_val.shape)
    for e in range(self.epochs):
      for i in range(0,x_train.shape[0],self.batch_size):
        x_batch = x_train[i:min(i+batch_size,x_train.shape[0])]
        y_batch = y_train[:,i:min(i+batch_size,x_train.shape[0])] # 10x32
        output = [0] * x_batch.shape[0]


        # forward prop  
        for j in range(x_batch.shape[0]):
          output[j] = np.expand_dims(x_batch[j], axis=0)
          for layer in self.CNN_layers:
            output[j] = layer.forward(output[j])
        output_array = self.flatten.forward(output)
        Y_hat = self.nn_layer.forward(output_array)


        # backprop
        gradY = self.nn_layer.backpropagation(y_batch)
        gradY = self.flatten.backpropagation(gradY)
        back_prop  = []
        for j in range(x_batch.shape[0]):
          back_prop.append(gradY[j])
          for layer in reversed(self.CNN_layers):
            back_prop.append(layer.backpropagation(back_prop[-1]))
      (loss_train, acuracy_train) = self.test(x_train, y_train)
      (loss_val,accuracy_val) = self.test(x_val,y_val)
      print("epochs: ", e, "loss_train: ", loss_train, "accuracy_train:", acuracy_train,
            "loss_val", loss_val, "accuracy_validation:", accuracy_val)


  def test(self, x_test, y_test):
        output = [0] * x_test.shape[0]
        for j in range(x_test.shape[0]):
          output[j] = np.expand_dims(x_test[j], axis=0)
          for layer in self.CNN_layers:
            output[j] = layer.forward(output[j])
        output_array = self.flatten.forward(output)
        Y_hat = self.nn_layer.forward(output_array)
        loss = self.nn_layer.cost(y_test,self.nn_layer.loss)
        accuracy = self.nn_layer.get_accuracy(np.argmax(Y_hat,0),np.argmax(y_test,0))
        return (loss, accuracy)


