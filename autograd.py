
import numpy as np

class Tensor:
    def __init__(self, data, depends_on = None, requires_grad=False, operator = ""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self.depends_on = depends_on
        self.operator = operator
        self._backward = None

    def __add__(self, other):
        result = Tensor(self.data + other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "+")
        def _backward(grad):
            new = grad + 0
            if self.requires_grad:
                self.backward(new)
            new = grad + 0
            if other.requires_grad:
                other.backward(new)
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
        result = Tensor(self.data.dot(other.data), depends_on=[self, other], requires_grad=self.requires_grad or other.requires_grad, operator="dot")

        def _backward(grad):
            if self.requires_grad:
                self.backward(grad.dot(other.data.T))
            if other.requires_grad:
                other.backward(self.data.T.dot(grad))


            # if self.requires_grad:
            #     self.backward(grad.dot(other.data.T))
            # if other.grad:
            #     grad.backward(grad.dot(self.data))

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

    def softmax(self):
        exp_data = np.exp(self.data - np.max(self.data))
        result = Tensor(exp_data / np.sum(exp_data), depends_on=[self], requires_grad=self.requires_grad, operator="softmax")


        return result

    def backward(self,backward_grad = None):

        if backward_grad is None:
            print("1111.1",self.data, self.grad, backward_grad)
            backward_grad = np.ones_like(self.data)
            print("1111.2",self.data, self.grad, backward_grad)


        print("3333.1",self.data, self.grad, backward_grad)
        self.grad += backward_grad
        print("3333.2",self.data, self.grad, backward_grad)

        if self.depends_on is not None:
            self._backward(backward_grad)

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

a = Tensor(2, requires_grad=True)
b = Tensor(6, requires_grad=True)
c = a + b * a

d = a *b
e = a - b

c.backward()
d.backward()
e.backward()

print("Gradient của a:")
print(a.grad)
print("Gradient của b:")
print(b.grad)

a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[2, 0], [1, 2]], requires_grad=True)
c = a + b
d = c * a
c.backward()

print("Gradient của a:")
print(a.grad)
print("Gradient của b:")
print(b.grad)

y_true = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
y_pred = Tensor([1.1, 1.9, 3.2, 3.7], requires_grad=True)

# Hàm loss: MSE (Mean Squared Error)
loss = ((y_pred - y_true) ** Tensor(2)).mean()

print("loss:", loss)
# Tính gradient
loss.backward()

print("Gradient của y_pred:")
print(y_pred.grad)

print("Gradient của y_true:")
print(y_true.grad)