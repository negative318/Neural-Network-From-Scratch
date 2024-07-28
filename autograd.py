
import numpy as np

class Tensor:
    def __init__(self, data, depends_on = None, requires_grad=False, operator = ""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.depends_on = depends_on
        self.operator = operator

    def __add__(self, other):
        return Tensor(self.data + other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "+")


    def __sub__(self, other):
        return Tensor(self.data - other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "-")

    def __mul__(self, other):
        return Tensor(self.data * other.data,depends_on = [self,other], requires_grad=self.requires_grad or other.requires_grad, operator = "*")


    def __truediv__(self,other):
        return Tensor(self.data / other.data, depends_on= [self, other], requires_grad=self.requires_grad or other.requires_grad, operator = "/")


    def __pow__(self,other):
        return Tensor(self.data ** other.data, depends_on= [self, other], requires_grad=self.requires_grad or other.requires_grad, operator = "**" )

    def mean(self):
        return Tensor(self.data.mean(), depends_on = [self], requires_grad=self.requires_grad, operator = "mean")


    def sum(self):
        return Tensor(self.data.sum(), depends_on=[self], requires_grad=self.requires_grad, operator="sum")


    def dot(self, other):
        return Tensor(self.data.dot(other.data), depends_on=[self, other], requires_grad=self.requires_grad or other.requires_grad, operator="dot")


    def sin(self):
        return Tensor(np.sin(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="sin")


    def cos(self):
        return Tensor(np.cos(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="cos")

    def relu(self):
        return Tensor(np.maximum(self.data, 0), depends_on=[self], requires_grad=self.requires_grad, operator="relu")


    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)), depends_on=[self], requires_grad=self.requires_grad, operator="sigmoid")


    def tanh(self):
        return Tensor(np.tanh(self.data), depends_on=[self], requires_grad=self.requires_grad, operator="tanh")


    def backward(self,backward_grad = None):
        if backward_grad is None:
            print("1111.1",self.data, self.grad, backward_grad)
            backward_grad = np.ones_like(self.data)
            print("1111.2",self.data, self.grad, backward_grad)
        if self.grad is None:
            print("2222.1",self.data,  self.grad, backward_grad)
            self.grad = backward_grad
            print("2222.2",self.data,  self.grad, backward_grad)
        else:
            print("3333.1",self.data, self.grad, backward_grad)
            self.grad += backward_grad
            print("3333.2",self.data, self.grad, backward_grad)
        if self.depends_on is not None:
            if self.operator == "+":
                new = backward_grad + 0
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(new)
                new = backward_grad + 0
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(new)

            elif self.operator == "-":
                new = backward_grad + 0
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(new)
                new = backward_grad + 0
                print(new)
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(-new)

            elif self.operator == "*":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad * self.depends_on[1].data)
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(backward_grad * self.depends_on[0].data)

            elif self.operator == "/":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad / self.depends_on[1].data)
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(backward_grad * (-self.depends_on[0].data / (self.depends_on[1].data ** 2)))

            elif self.operator == "**":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad * (self.depends_on[1].data * self.depends_on[0].data **  (self.depends_on[1].data-1))) # Added .data to self.depends_on[1]
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(backward_grad * (self.depends_on[0].data ** self.depends_on[1].data * np.log(self.depends_on[0].data)))

            elif self.operator == "mean":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad / self.data.size)


            elif self.operator == "sum":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad * np.ones_like(self.depends_on[0].data))


            elif self.operator == "dot":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad.dot(self.depends_on[1].data.T))
                #if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(self.depends_on[0].data.T.dot(backward_grad))
                if(self.depends_on[1].requires_grad == True): self.depends_on[1].backward(backward_grad.dot(self.depends_on[0].data))


            elif self.operator == "sin":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad * np.cos(self.depends_on[0].data))


            elif self.operator == "cos":
                if(self.depends_on[0].requires_grad == True): self.depends_on[0].backward(backward_grad * -np.sin(self.depends_on[0].data))


            elif self.operator == "relu":
                if self.depends_on[0].requires_grad: self.depends_on[0].backward(backward_grad * (self.depends_on[0].data > 0))


            elif self.operator == "sigmoid":
                if(self.depends_on[0].requires_grad == True):
                  sigmoid_data = 1 / (1 + np.exp(-self.depends_on[0].data))
                  self.depends_on[0].backward(backward_grad * sigmoid_data * (1 - sigmoid_data))


            elif self.operator == "tanh":
                if(self.depends_on[0].requires_grad == True):
                  tanh_data = np.tanh(self.depends_on[0].data)
                  self.depends_on[0].backward(backward_grad * (1 - tanh_data ** 2))
    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[2, 0], [1, 2]], requires_grad=True)
c = a.dot(b)

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