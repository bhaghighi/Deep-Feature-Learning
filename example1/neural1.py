# example: https://www.analyticsvidhya.com/blog/2016/04/neural-networks-python-theano/
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from random import random
#%%
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [0,0,0,1]
#%%
X = T.matrix("X")
#W = theano.shared(np.array([random(),random()]).astype("float32"))
#W = theano.shared(np.random.rand(2,).astype("float32"), name = "W1")
W = theano.shared(np.array([random(),random()]))
b = theano.shared(1.)
#%%
# forward propogation 
z = T.dot(X,W) + b
y_hat = 1/ (1 + T.exp(-z))
#y_hat = T.nnet.sigmoid(z)
forward = theano.function([X],y_hat)
print (forward(inputs),forward(inputs).shape)
#%%
# cost, gradient and optimizar
y = T.vector("y")
cost = -(y * T.log(y_hat) + (1-y) * T.log(1-y_hat)).sum()
loss = theano.function([X,y],cost)
print (loss(inputs,outputs),loss(inputs,outputs).shape)
#%%
#gradinets
learning_rate = 0.01
dW,db = T.grad(cost,[W,b])
train = theano.function(
        inputs = [X,y],
        outputs = [y_hat,cost],
        updates = [
                [W, W - learning_rate * dW],
                [b, b-learning_rate*db]
                ]
        )
#%%
cost = []
for i in range (3000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    
for i in range(len(inputs)):
    print ('The output for x1=%d | x2=%d is %.2f' % (inputs[i][0],inputs[i][1],pred[i]))
plt.plot(cost)