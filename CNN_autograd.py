from activateFuncion import *
from autograd import *
import numpy as np

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

    self.kernel = Tensor(np.random.randn(*self.kernel_shape), requires_grad= True)
    self.bias = Tensor(np.random.randn(*self.bias_shape), requires_grad= True)

    self.l_rate = l_rate
    self.active_text = activeFuncion



      
    

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


  def forward(self, input):
    self.input = input
    output = self.input.conv(self.kernel)
    return self.active(output)


  def update_parameter(self):
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

#input 4 chiều là 1 tensor
class Flattening():
  def __init__(self):
    pass
  def forward(self, input):
    output = input.flatten()
    return output.T


