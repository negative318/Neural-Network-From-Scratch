
from MLP import *

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size,np.max(Y) + 1))
  one_hot_Y[np.arange(Y.size),Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

num_test = 1000
y_train = one_hot(y_train)
y_test = one_hot(y_test)
x_train = x_train/255
x_test = x_test/255
x_train = x_train[0:num_test]
y_train = y_train[:,0:num_test]
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

class Convolutional:


  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def __init__(self, input_shape, kernel_size, output_depth):
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

  # input (Di,Hi,Wi)
  # img (Hi, Wi)
  # kernel(Hk,Wk)
  def conv(self, image, kernel, status):
    image = np.array(image)
    kernel = np.array(kernel)
    img_shape = image.shape
    kernel_shape = kernel.shape

    if status == "forward":
      out_put_height = img_shape[0]-kernel_shape[0]+1
      out_put_width = img_shape[1]-kernel_shape[1]+1
      output = np.zeros((out_put_height, out_put_width))
      for i in range (out_put_height):
        for j in range (out_put_width):
          output[i][j] = np.sum(image[i:i+kernel_shape[0],j:j + kernel_shape[1]]*kernel)
      return output

    if status == "backprop":

      out_put_height = img_shape[0]+kernel_shape[0]-1
      out_put_width = img_shape[1]+kernel_shape[1]-1
      output = np.zeros((out_put_height, out_put_width))

      input_height, input_width = image.shape
      input_height = input_height + (kernel_shape[0]-1)*2
      input_width = input_width + (kernel_shape[1]-1)*2

      input = np.zeros((input_height, input_width))
      for i in range (input_height):
        for j in range (input_width):
          if i < kernel_shape[0]-1 or j < kernel_shape[1]-1 or i > input_height-kernel_shape[0] or j > input_width-kernel_shape[1]:
            input[i][j] = 0
          else:
            input[i][j] = image[i-kernel_shape[0]+1][j-kernel_shape[1]+1]
      for i in range (out_put_height):
        for j in range (out_put_width):
          output[i][j] = np.sum(input[i:i+kernel_shape[0],j:j + kernel_shape[1]]*kernel)
      return output

  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def forward(self, input):
    self.input = input
    self.output = np.zeros(self.output_shape)
    for j in range (self.output_shape[0]):
      for i in range (self.input_shape[0]):
          self.output[j] += self.conv(self.input[i], self.kernel[j][i],"forward") + self.bias[j]

    return self.output


  #gradY (Dy, Hy, Wy)
  # input (Di,Hi,Wi)
  # output (Do,Ho,Wo)
  # kernel (Do,Di,Hk,Wk)
  # bias (Do,Ho,Wo)
  def backpropagation(self, gradY, eta):
    grad_kernel = np.zeros(self.kernel_shape)
    grad_input = np.zeros(self.input_shape)
    for i in range (self.output_shape[0]):
      for j in range(self.input_shape[0]):
        grad_input[j] += self.conv(gradY[i], self.kernel[i][j],"backprop")
        grad_kernel[i][j] = self.conv(self.input[j], gradY[i],"forward")
    self.kernel -= eta*grad_kernel
    self.bias -= eta*gradY
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

test = np.random.rand(35,1)

epochs = 30
batch_size = 32
CNN1 = Convolutional((1,28,28),3,8)
pool1 = MaxPoolingLayer(2)
CNN2 = Convolutional((8,13,13),3,16)
pool2 = MaxPoolingLayer(2)
flatten = Flattening()
nn = NeuralNetwork(layers_size=[576,10],activations = ["", "softmax"], loss = "crossEntropy", l_rate = 1e-6) # 16*6*6 = 576

print(x_train.shape[0])
for e in range(epochs):
  for i in range(0,x_train.shape[0],batch_size):
      x_batch = x_train[i:min(i+batch_size,x_train.shape[0])]
      y_batch = y_train[:,i:min(i+batch_size,x_train.shape[0])] # 10x32
      output = [0] * x_batch.shape[0]


      # forward prop
      for j in range(x_batch.shape[0]):
        x_batch_reshaped = np.expand_dims(x_batch[j], axis=0)
        output[j] = CNN1.forward(x_batch_reshaped)
        output[j] = pool1.forward(output[j])
        output[j] = CNN2.forward(output[j])
        output[j] = pool2.forward(output[j])
      output_array = flatten.forward(output)
      Y_hat = nn.forward(output_array)
      if(i%10 == 0):
        print("epochs: ", e, "interation:", i, "loss: ", nn.cost(y_batch,nn.loss), "accuracy:", nn.get_accuracy(np.argmax(Y_hat,0),np.argmax(y_batch,0)))

      # backprop
      gradY = nn.backpropagation(y_batch)
      gradY = flatten.backpropagation(gradY)
      back_prop  = []
      for j in range(x_batch.shape[0]):
        back_prop.append(gradY[j])
        back_prop.append(pool2.backpropagation(back_prop[-1]))
        back_prop.append(CNN2.backpropagation(back_prop[-1],1e-6))
        back_prop.append(pool1.backpropagation(back_prop[-1]))
        back_prop.append(CNN1.backpropagation(back_prop[-1],1e-6))