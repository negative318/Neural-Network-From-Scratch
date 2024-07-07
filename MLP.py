import numpy as np
import matplotlib.pyplot as plt
import keras



N = 3

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

# for i in range(10):
#   plt.imshow(x_train[y_train == i][0], cmap ='gray')
#   plt.title("label: {}".format(i), fontsize=16)
#   plt.show()

x_train = x_train /255.0
x_test = x_test /255.0

print(y_test)
print(y_train)
print(y_test.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0],-1).T
x_test = x_test.reshape(x_test.shape[0],-1).T
print(x_test.shape)
print(x_train.shape)




def init_parameter(N):
  d = [10] * (N+1)
  d[0] = 784
  W = []
  b = []
  for i in range(N):
    W.append(np.random.randn(d[i],d[i+1]))
    b.append(np.random.randn(d[i+1],1))
  return W,b

def RELU(Z):
  return np.maximum(0,Z)

def softmax(Z):
  Z_max = np.max(Z)
  exp_Z = np.exp(Z - Z_max)
  A = exp_Z / np.sum(exp_Z)
  return A



def one_hot(Y):
  one_hot_Y = np.zeros((Y.size,Y.max()+1))
  one_hot_Y[np.arange(Y.size),Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def prop(W,b,X,N):
  Z = []
  A = []
  for i in range(N):
    if i == 0 :
      Z.append(W[0].T.dot(X) + b[0])
      A.append(RELU(Z[-1]))
      print(1,A)
    elif i != N-1 :
      Z.append(W[i].T.dot(A[i-1]) + b[i])
      A.append(RELU(Z[-1]))
      print(2,A)
    else:
      Z.append(W[i].T.dot(A[i-1])+ b[i])
      print(3,Z)
      A.append(softmax(Z[-1]))
      print(3,A)
  return Z,A
def back_prop(Z,A,W,X,Y,N):
  E = []
  dW = []
  db = []
  one_hot_Y = one_hot(Y)
  for i in range(N-1, -1, -1):
    if (i == 0):
      E.append(W[i+1].dot(E[-1]))
      E[-1][Z[0] <= 0] = 0
      dW.insert(0,X.dot(E[-1].T))
    elif(i == N-1):
      E.append(1/Y.size * (A[-1] - one_hot_Y))
      dW.insert(0,A[i-1].dot(E[-1].T))

    else:
      E.append(W[i+1].dot(E[-1]))
      E[-1][Z[i] <= 0] = 0
      dW.insert(0,A[i-1].dot(E[-1].T))

    db.insert(0,np.sum(E[-1],axis=1,keepdims=True))


  return dW,db
def update_parameter(W,b,dW,db,eta):

  for i in range(len(W)):
    W[i] = W[i] - eta * dW[i]
    b[i] = b[i] - eta * db[i]
  return W,b
def cost(Y, Yhat):
  return -np.sum(Y * np.log(Yhat))/Y.size


def get_predictions(Yhat):
  return np.argmax(Yhat,0)
def get_accuracy(predictions,Y):
  return np.sum(predictions == Y) / Y.size


#N là số layer
def train(N,W,b,X,Y,eta,epochs):
  for i in range(epochs):
    Z,A = prop(W,b,X,N)
    dW,db = back_prop(Z,A,W,X,Y,N)
    W,b = update_parameter(W,b,dW,db,eta)
    #if i%100 == 0:
    print(i, "cost: ", cost(Y,A[-1]), "Accuracy", get_accuracy(get_predictions(A[-1]),Y))
  return W,b

def test(N,W,b,X,Y):
    one_hot_Y = one_hot(Y)
    Z,A = prop(W,b,X,N)
    print("cost: ", cost(one_hot_Y,A[-1]), get_accuracy(get_predictions(A[-1]),Y))
    return get_predictions(A[-1])



W,b = init_parameter(N)
W,b = train(N,W,b,x_train,y_train,0.01,5001)


Yhat = test(N,W,b,x_test,y_test)
print(Yhat)
print(y_test)
print("Accuracy: ",get_accuracy(Yhat,y_test)*100)