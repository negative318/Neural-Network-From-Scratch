from activateFuncion import *
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

    self.kernel = np.random.randn(*self.kernel_shape)
    self.bias = np.random.randn(*self.bias_shape)

    self.l_rate = l_rate
    self.active_text = activeFuncion
    self.activeFunction = None
    if(activeFuncion == "relu"):
      self.activeFuncion = ReLU()
      self.relu_array = []
    elif(activeFuncion == "sigmoid"):
      self.activeFuncion = sigmoid()
    elif(activeFuncion == "tanh"):
      self.activeFuncion = tanh()

    
  # input (Di,Hi,Wi)
  # img (Hi, Wi)
  # kernel(Hk,Wk)
  def conv(self, image, kernel, padding):
    image = np.array(image)
    kernel = np.array(kernel)
    img_shape = image.shape
    kernel_shape = kernel.shape
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    out_put_height = padded_image.shape[0] - kernel_shape[0] + 1
    out_put_width = padded_image.shape[1] - kernel_shape[1] + 1
    output = np.zeros((out_put_height, out_put_width))
    for i in range (out_put_height):
      for j in range (out_put_width):
        output[i][j] = np.sum(padded_image[i:i+kernel_shape[0],j:j + kernel_shape[1]]*kernel)
    return output


  
  def active(self,input):
      if self.active_text == "":
        return input
      if(self.active_text == "relu"):
        self.relu_array.append(input)
      return self.activeFuncion.forward(input)

  def derivative(self,output):
    if(self.active_text ==""):
        return output
    if(self.active_text =="relu"):
      return output * self.activeFuncion.backpropagation(self.relu_array.pop(0))
    return self.activeFuncion.backpropagation(output)
  

  
  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def forward(self, input):
    self.input = np.array(input)
    self.output = np.zeros(self.output_shape)
    for j in range (self.output_shape[0]):
      for i in range (self.input_shape[0]):
          self.output[j] += self.conv(self.input[i], self.kernel[j][i], padding = 0)
      self.output[j] += self.bias[j]
    self.output = self.active(self.output)
    return self.output


  #gradY (Dy, Hy, Wy)
  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def backpropagation(self, gradY):
    gradY = self.derivative(gradY)
    grad_kernel = np.zeros(self.kernel_shape)
    grad_input = np.zeros(self.input_shape)



# xoay 180
    root_kernel = np.zeros(self.kernel_shape)

    for i in range (self.output_shape[0]):
      for j in range(self.input_shape[0]):
        root_kernel[i][j] = np.rot90(self.kernel[i][j], 2)
        grad_input[j] += self.conv(gradY[i], root_kernel[i][j], padding = root_kernel[i][j].shape[0] - 1)
        grad_kernel[i][j] = self.conv(self.input[j], gradY[i], padding = 0)
    self.kernel -= self.l_rate*grad_kernel
    self.bias -= self.l_rate*gradY
    return grad_input


class MaxPoolingLayer:
  def __init__(self, pool_size):
    self.pool_size = pool_size

  def forward(self, input): # input 3 chiều
    self.input = np.array(input)
    out_depth = self.input.shape[0]
    out_height = int(self.input.shape[1]/self.pool_size)
    out_width = int(self.input.shape[2]/self.pool_size)
    if(self.input.shape[1] % self.pool_size != 0):
      out_height += 1
    if(self.input.shape[2] % self.pool_size != 0):
      out_width += 1
    self.output = np.zeros((out_depth, out_height, out_width))
    for d in range(out_depth):
      for i in range(0, out_height):
        for j in range(0, out_width):
          start_row = i * self.pool_size
          end_row = min(start_row + self.pool_size, self.input.shape[1])
          start_col = j * self.pool_size
          end_col = min(start_col + self.pool_size, self.input.shape[2])
          if start_row < end_row and start_col < end_col:
              self.output[d][i][j] = np.max(self.input[d,start_row:end_row, start_col:end_col])
    return self.output
  def backpropagation(self, gradY):
    grad_input = np.zeros(self.input.shape)
    for d in range(gradY.shape[0]):
      for i in range(0, gradY.shape[1]):
        for j in range(0, gradY.shape[2]):
          start_row = i * self.pool_size
          end_row = min(start_row + self.pool_size, self.input.shape[1])
          start_col = j * self.pool_size
          end_col = min(start_col + self.pool_size, self.input.shape[2])
          if start_row < end_row and start_col < end_col:
            max_val = np.max(self.input[d,start_row:end_row, start_col:end_col])
            max_indices = np.where(self.input[d,start_row:end_row, start_col:end_col] == max_val)
            max_indices = (max_indices[0] + start_row, max_indices[1] + start_col)
            grad_input[d][max_indices[0][0]][max_indices[1][0]] = gradY[d][i][j]
    return grad_input

class Flattening():
  def __init__(self):
    pass
  def forward(self, input):
    input = np.array(input)
    self.input = input
    self.output = input.reshape(input.shape[0], -1)
    return self.output.T
  def backpropagation(self, gradY):
    return gradY.T.reshape(self.input.shape)

class Model:
  def __init__(self, CNN_layers, nn_layer):
    self.CNN_layers = CNN_layers
    self.nn_layer = nn_layer
    self.flatten = Flattening()
  def train(self, x_train,y_train,x_val,y_val, batch_size, epochs):
    self.batch_size = batch_size
    self.epochs = epochs
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
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
      print("loss_train: ", loss_train, "accuracy_train:", acuracy_train)
      # (loss_train, acuracy_train) = self.test(x_train, y_train)
      # (loss_val,accuracy_val) = self.test(x_val,y_val)
      # print("epochs: ", e, "loss_train: ", loss_train, "accuracy_train:", acuracy_train,
      #       "loss_val", loss_val, "accuracy_validation:", accuracy_val)


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


