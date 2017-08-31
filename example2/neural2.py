#%%
import os 
import numpy as np 
import theano 
import theano.tensor as T
import matplotlib.pyplot as plt 
#%%
# Data and normalization 
inputs = np.array([[3,5],[5,1],[10,2]]).astype("float32")
outputs = np.array([[75],[82],[93]]).astype("float32")
inputs = inputs / np.amax(inputs, axis = 0)
outputs = outputs / 100
#%%
# hyper parameters and weights
inputLayerSize = 2
hiddenLayerSize = 3
outputLayerSize = 1
W1 = theano.shared((np.random.randn(inputLayerSize,hiddenLayerSize)).astype("float32"), name = "W1")
b1 = theano.shared( np.zeros(hiddenLayerSize).astype("float32") , name = "b1")
W2 = theano.shared((np.random.randn(hiddenLayerSize,outputLayerSize)).astype("float32"), name = "W2")
b2 = theano.shared( np.zeros(outputLayerSize).astype("float32") , name = "b2")
#%%
# Forward propogation 
X = T.matrix("X") 
z1 = T.dot(X,W1) + b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1,W2) + b2
# using ReLu improve the results a lot 
y_hat = T.nnet.relu(z2)
forward = theano.function([X], y_hat)
#%%
# cost function, gradient and optimizer
epsilon = 0.01
y = T.fcol("y")
loss = 0.5 * ((y - y_hat)**2).sum()
calloss = theano.function([X,y], loss)
# gradinet 
dW1, dW2 = T.grad(loss, [W1,W2])
db1, db2 = T.grad(loss,[b1,b2])
# optimizer 
#%%       
train  = theano.function(
        inputs = [X,y],
        outputs = [y_hat,loss],
        updates = [
                [W2, W2 - epsilon * dW2],
                [W1, W1 - epsilon * dW1],
                [b2, b2 - epsilon * db2],
                [b1, b1 - epsilon * db1]
                ]
        )
#%%
cost = []
for i in range(20000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
plt.plot(cost)
print (pred,"\n" ,outputs)
    
    
    
    
    
    
    
    
    
    
    
    