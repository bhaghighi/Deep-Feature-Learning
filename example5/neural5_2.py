# Iris data set with theano 
#%%
import pandas as pd 
import numpy as np
import theano 
import theano.tensor as T
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%
# Download the data and normalization 
df = pd.read_csv("https://archive.ics.uci.edu//ml//machine-learning-databases//iris//iris.data", header = None)
df.columns = ["f1","f2","f3","f4","lables"]
# explore the data before processing. could you say which features are important? 
#sns.pairplot(df, hue="lables")
# convert data frame to numpy array and then seperate data from labels
# get one_hot encoding with pandas data frame 
one_hot = pd.get_dummies(df.lables)
# drop the labels column (we don't need it anymore)
df= df.drop("lables", axis = 1)
df = df.join(one_hot)
# cross validation 
train, test = train_test_split(df, test_size = 0.3)
# we have to convert data frame to numpy array to work with thenao.
train = train.values
test = test.values
print (type(df))
train_X = train[:,0:4].astype("float32")
train_Y = train[:,4:].astype("float32")
test_X = test[:,0:4].astype("float32")
test_Y = test[:,4:].astype("float32")

#%%
# defining hyper parameters for the network
inputLayerSize = 4
hiddenLayerSize = 256
outputLayerSize = 3 
epsilon = 0.01 
# 
W1 = theano.shared((np.random.randn(inputLayerSize,hiddenLayerSize)).astype("float32"), name = "W1")
b1 = theano.shared(np.zeros(hiddenLayerSize).astype("float32"), name = "b1")
W2 = theano.shared((np.random.randn(hiddenLayerSize, outputLayerSize)).astype("float32"), name = "W2")
b2 = theano.shared(np.zeros(outputLayerSize).astype("float32"), name = "b2")
#%%
# Forward propogation 
X = T.fmatrix("X")
z1 = T.dot(X, W1) + b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1, W2) + b2
y_hat = T.nnet.softmax(z2)
forward = theano.function([X], y_hat)
#%%
# backward propogation 
y = T.fmatrix("y")
# sum of square cost function 
#loss = 0.5 * ((y - y_hat)**2).sum()
# cross entropy cost function
# type of cost function is really important. wiht SQ did not reach good accuracy
# but with cross entropy accuray is about 0.986
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() 
calloss = theano.function([X,y], loss)
# gradinet 
dW1, dW2 = T.grad(loss, [W1,W2])
db1, db2 = T.grad(loss,[b1,b2])
#%%
# train 
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
for i in range(10000):
    pred, cost_iter = train(train_X, train_Y)
    cost.append(cost_iter)
plt.plot(cost)
#print (pred,"\n" ,train_Y)
# calculate accuray 
train_accuracy = np.mean(np.argmax(train_Y, axis=1) == np.argmax(pred, axis=1))
print (train_accuracy)
#
# test on testion set
cost = []
for i in range(10000):
    pred, cost_iter = train(test_X, test_Y)
    cost.append(cost_iter)
plt.plot(cost)
#print (pred,"\n" ,train_Y)
# calculate accuray 
train_accuracy = np.mean(np.argmax(test_Y, axis=1) == np.argmax(pred, axis=1))
print (train_accuracy)








