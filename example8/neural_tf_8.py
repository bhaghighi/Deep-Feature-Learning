# Iris using tensorflow 
#%%
# import neccessary packages
import os 
import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# use only CPU 
os.environ["CUDA_VISIBLE_DEVICES"]=""
#%%
# get the neccessary data and normalize and hot encodin it 
iris = pd.read_csv("https://archive.ics.uci.edu//ml//machine-learning-databases//iris//iris.data", header = None)
iris.columns = ["f1","f2","f3","f4", "lable"]
# hot encoding 
df = pd.get_dummies(iris, columns = ["lable"])
# seperate train and lables 
inputs = df.iloc[:,0:4].values
outputs = df.iloc[:,4:].values
#%%
# hyper paramaeters of network
inputSize = 4
hiddenSize = 256
outputSize = 3
# input, output, weight and bios
X = tf.placeholder(tf.float32,[None,4])
y = tf.placeholder(tf.float32,[None,3])
W1 = tf.Variable(np.random.randn(inputSize,hiddenSize), dtype=tf.float32)
b1 = tf.Variable(np.zeros(hiddenSize), dtype = tf.float32)
W2 = tf.Variable(np.random.randn(hiddenSize,outputSize), dtype=tf.float32)
b2 = tf.Variable(np.zeros(outputSize), dtype=tf.float32)
#%%
# forward propogation 
z1 = tf.matmul(X,W1) + b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1,W2) + b2
#y_hat = tf.nn.softmax(z2)
# if you use tf.nn.softmax_cross_entropy_with_logits, there is no need to apply softmax on output layer
y_hat = z2
# initialization and session start
# initialize weights 
init = tf.global_variables_initializer() # you need also start session to initialize it!
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print (sess.run(init))
print(sess.run(y_hat, {X: inputs }))
#%%
# cost and optimizer 
#loss = tf.reduce_sum(tf.square(y_hat - y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
#%%
# train
cost = []

for i in range(10000):
    sess.run(train,{X:inputs, y:outputs})
    cost.append(sess.run(loss, {X:inputs,y:outputs}))

predict = sess.run(y_hat, {X: inputs})
print ("prediction", predict)
print ("loss", sess.run(loss, {X:inputs, y:outputs}))

# finding accuracy
train_accuracy = np.mean(np.argmax(outputs, axis=1) == np.argmax(predict, axis=1))
print (train_accuracy)
print ("loss", sess.run(loss, {X:inputs, y:outputs}))
plt.plot(cost)








