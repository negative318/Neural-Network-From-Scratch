
import numpy as np
from scipy.signal import correlate2d

class Tensor:
    def __init__(self, data, depends_on = None, requires_grad=False, operator = ""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self.depends_on = depends_on
        self.operator = operator
        self._backward = None



    @property
    def T(self):
        result = Tensor(self.data.T, requires_grad=self.requires_grad, depends_on=[self], operator="transpose")
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad.T)
        
        result._backward = _backward
        return result    


    def __add__(self, other):
        result = Tensor(self.data + other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "+")
        def _backward(grad):
            new_self_grad = grad + 0
            new_other_grad = grad + 0
            
            if self.requires_grad:
                # print("aaaaaaa1", self.data.shape, grad.shape)
                if self.data.shape != grad.shape:
                    # print("aaaaaaaaaaaaaa2", self.data.shape, grad.shape)
                    new_self_grad = np.sum(grad, axis=0, keepdims=True)
                    new_self_grad = new_self_grad.reshape(self.data.shape)
            self.backward(new_self_grad)
            
            if other.requires_grad:
                # print("bbbbbbbbbbb1", other.data.shape, grad.shape)
                if other.data.shape != grad.shape:
                    # print("bbbbbbbbbbb2", other.data.shape, grad.shape)
                    new_other_grad = np.sum(grad, axis=0, keepdims=True)
                    # print(new_other_grad.shape)
                    new_other_grad = new_other_grad.reshape(other.data.shape)
            other.backward(new_other_grad)
        result._backward = _backward
        return result

    def __sub__(self, other):
        result = Tensor(self.data - other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "-")
        def _backward(grad):
            new = grad + 0
            if self.requires_grad:
              self.backward(new)
            new = grad + 0
            if other.requires_grad:
              other.backward(-new)
        result._backward = _backward
        return result

    def __mul__(self, other):
        result = Tensor(self.data * other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "*")
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad * other.data)
            if other.requires_grad:
                other.backward(grad * self.data)
        result._backward = _backward
        return result



    def __truediv__(self,other):
        result = Tensor(self.data / other.data, depends_on= [self, other], requires_grad=self.requires_grad or other.requires_grad, operator = "/")
        def _backward(grad):
          if self.requires_grad:
              self.backward(grad / other.data)
          if other.requires_grad:
              other.backward(grad * (-self.data / (other.data ** 2)))
        result._backward = _backward
        return result

    def __pow__(self,other):
        result = Tensor(self.data ** other.data, depends_on= [self, other], requires_grad=self.requires_grad or other.requires_grad, operator = "**" )
        def _backward(grad):
          if self.requires_grad:
              self.backward(grad * (other.data * self.data **  (other.data-1)))
          if other.requires_grad:
              other.backward(grad * (self.data ** other.data * np.log(self.data)))
        result._backward = _backward
        return result

    def mean(self):
        result = Tensor(self.data.mean(), depends_on = [self], requires_grad=self.requires_grad, operator = "mean")
        def _backward(grad):
            if self.requires_grad:
              self.backward(grad / self.data.size)
        result._backward = _backward
        return result


    def sum(self):
        result = Tensor(self.data.sum(), depends_on=[self], requires_grad=self.requires_grad, operator="sum")
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad * np.ones_like(self.data))
        result._backward = _backward
        return result
    
    

    def dot(self, other):
        result = Tensor(np.dot(self.data,other.data), depends_on=[self, other], requires_grad=self.requires_grad or other.requires_grad, operator="dot")

        def _backward(grad):
            
            if self.requires_grad:
                self_grad = np.dot(grad, np.transpose(other.data))
                self.backward(self_grad)
            if other.requires_grad:
                other_grad = np.dot(np.transpose(self.data), grad)
                other.backward(other_grad)
        result._backward = _backward
        return result

    def sin(self):
        result = Tensor(np.sin(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="sin")

        def _backward(grad):
           if self.grad:
              self.backward(grad * np.cos(self.data))

        result._backward = _backward
        return result


    def cos(self):
        result = Tensor(np.cos(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="cos")

        def _backward(grad):
            if self.requires_grad:
              self.backward(grad * -np.sin(self.data))

        result._backward = _backward
        return result

    def exp(self):
        result = Tensor(np.exp(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="exp")
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad * np.exp(self.data))
        result._backward = _backward
        return result

    def relu(self):
        result = Tensor(np.maximum(self.data, 0), depends_on=[self], requires_grad=self.requires_grad, operator="relu")

        def _backward(grad):
            if self.requires_grad:
              self.backward(grad * (self.data > 0))

        result._backward = _backward
        return result
    def sigmoid(self):
        result = Tensor(1 / (1 + np.exp(-self.data)), depends_on=[self], requires_grad=self.requires_grad, operator="sigmoid")

        def _backward(grad):
            if self.requires_grad == True:
                  sigmoid_data = 1 / (1 + np.exp(-self.data))
                  self.backward(grad * sigmoid_data * (1 - sigmoid_data))

        result._backward = _backward
        return result

    def tanh(self):
        result = Tensor(np.tanh(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="tanh")

        def _backward(grad):
          if self.requires_grad:
            self.backward(grad * (1 - np.tanh(self.data) ** 2))

        result._backward = _backward
        return result


    def mse_loss(self, target):
        diff = (self - target) ** Tensor(2)
        loss = diff.mean()
        def _backward(grad):
            if diff.requires_grad:
                diff.backward(grad / self.data.size)
        loss._backward = _backward
        return loss



    def abs(self):
        result = Tensor(np.abs(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="abs")
        def _backward(grad):
            if self.requires_grad:
                self.grad += grad * np.sign(self.data)
        result._backward = _backward
        return result
    def mae_loss(self, target):
        diff = (self - target).abs()
        loss = diff.mean()
        def _backward(grad):
            if diff.requires_grad:
                diff.backward(grad / self.data.size)
        loss._backward = _backward
        return loss


    def softmax(self):
        # # print(self.data.shape)
        # self_max = np.max(self.data, axis=0, keepdims=True)
        # # print(self.data - self_max)
        # exp_self = np.exp(self.data - self_max)
        # A = exp_self / np.sum(exp_self, axis=0, keepdims=True)
    
        # result = Tensor(A, depends_on=[self], requires_grad=self.requires_grad, operator="softmax")


        exp_self = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))  # Calculate softmax more stably
        result = Tensor(exp_self / np.sum(exp_self, axis=1, keepdims=True), depends_on=[self], requires_grad=self.requires_grad, operator="softmax")
        # print(result.data.shape)

        def _backward(grad):
          if self.requires_grad:
            self.grad += (result.data - grad) / grad.size
            self.backward(self.grad)
        result._backward = _backward
        
        return result

    


    #self = input = (num,chanel_in, h,w) = (64,3,32, 32)
    #other = kernel = (o_chanel, i_chanel,h,w) = (4,3,3,3)
    #output = (num, o_chanel, h,w) = (64,4,15,15)
    def conv(self, other,stride = 1, padding = 0):
        output_shape = (
            self.data.shape[0],
            other.data.shape[0], 
            self.data.shape[2] - other.data.shape[2] + 2 * padding + 1, 
            self.data.shape[3] - other.data.shape[3] + 2 * padding + 1
        )
        output = np.zeros(output_shape)
        # print(self.data.shape, other.data.shape, output_shape)
        padded_data = np.pad(self.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        for n in range(output.shape[0]):
            for j in range(output.shape[1]):
                for i in range(self.data.shape[1]):
                    output[n][j] += correlate2d(padded_data[n][i], other.data[j][i], mode='valid')[::stride, ::stride]
        result = Tensor(output,depends_on=[self,other], requires_grad = self.requires_grad or other.requires_grad, operator ="convolution")

        def _backward(grad):
          if self.requires_grad:
            flipped_kernel = np.flip(other.data, axis=(2, 3))
            grad_padded = np.pad(grad, ((0, 0), (0, 0), (other.data.shape[2] - 1, other.data.shape[2] - 1), (other.data.shape[3] - 1, other.data.shape[3] - 1)), mode='constant')
            grad_input = np.zeros_like(self.data)
            for n in range(output.shape[0]):
                for i in range(output.shape[1]):
                    for j in range(self.data.shape[1]):
                        grad_input[n][j] += correlate2d(grad_padded[n][i],flipped_kernel[i][j],mode='valid')
            self.backward(grad_input)
          if other.requires_grad:
            grad_kernel = np.zeros_like(other.data)
            for n in range(output.shape[0]):
                for i in range(output.shape[1]):
                    for j in range(self.data.shape[1]):
                        grad_kernel[i][j] += correlate2d(padded_data[n][j], grad[n][i], mode='valid')
            other.backward(grad_kernel)        
        result._backward = _backward
        return result

    def maxpooling(self, pool_size): # (num,chanel, hight,width)
        num, chanel, h, w = self.data.shape
        out_h = int(self.data.shape[2]/pool_size)
        out_w = int(self.data.shape[3]/pool_size)
        if(self.data.shape[2] % pool_size != 0):
            out_h += 1
        if(self.data.shape[3] % pool_size != 0):
            out_w += 1
        output = np.zeros((num,chanel,out_h,out_w))
        for n in range(num):
            for c in range(chanel):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * pool_size
                        h_end = min(h_start + pool_size, self.data.shape[2])
                        w_start = j * pool_size
                        w_end = min(w_start + pool_size, self.data.shape[3])
                        if h_start < h_end and w_start < w_end:
                            output[n, c, i, j] = np.max(self.data[n, c, h_start:h_end, w_start:w_end])
        result = Tensor(output, requires_grad=self.requires_grad, depends_on=[self], operator="maxpool")
        def _backward(grad):
            
            if self.requires_grad:
                grad_input = np.zeros_like(self.data)
                for n in range(num):
                    for c in range(chanel):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_start = i * pool_size
                                h_end = min(h_start + pool_size, self.data.shape[2])
                                w_start = j * pool_size
                                w_end = min(w_start + pool_size, self.data.shape[3])
                                max_val = np.max(self.data[n, c, h_start:h_end, w_start:w_end])
                                for h in range(h_start, h_end):
                                    for w in range(w_start, w_end):
                                        if self.data[n, c, h, w] == max_val:
                                            grad_input[n, c, h, w] += grad[n, c, i, j]
                                            
                self.backward(grad_input)

        result._backward = _backward
        return result


    def flatten(self):
        input_shape = self.data.shape
        output = self.data.reshape(input_shape[0],-1)
        result = Tensor(output, requires_grad= self.requires_grad, depends_on=[self], operator="flatten")
        def _backward(grad):
            if self.requires_grad:
                grad_input = grad.reshape(input_shape)
                self.backward(grad_input)
        result._backward =_backward
        return result




    def backward(self,backward_grad = None):

        if backward_grad is None:
            backward_grad = np.ones_like(self.data)
        self.grad += backward_grad
        if self.depends_on is not None:
            self._backward(backward_grad)

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'
