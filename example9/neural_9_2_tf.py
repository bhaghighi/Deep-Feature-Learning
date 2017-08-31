# MNIST with tensorflow (no hidden layer)
#%%
import os 
import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data
os.environ["CUDA_VISIBLE_DEVICES"]=""
import time
#%%
mnist = input_data.read_data_sets("/data1/bhaghighi/Data/DeepLearning/example9/MNIST_data/",one_hot=True)
X_train, y_train = mnist.train.images, mnist.train.labels
print (X_train.shape, y_train.shape)
#%%
# Design the network and hyperparameters
inputLayerSize = 784
outputLayerSize = 10
X = tf.placeholder(tf.float32, [None,784])
y =  tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([inputLayerSize,outputLayerSize]), dtype=tf.float32)
b = tf.Variable(tf.zeros([outputLayerSize]), dtype=tf.float32)
#%%
# Forward propogation 
z1 = tf.matmul(X,W) + b
y_hat = z1 # without applying softmax
# initialization of weights
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)
print (sess.run(y_hat, feed_dict={X: X_train}))
#%%
# cost function and optimization 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
cost = []
start = time.time()
for i in range(1000):
    sess.run(train, feed_dict={X:X_train,y:y_train})
    cost.append(sess.run(loss,feed_dict={X:X_train,y:y_train}))
end = time.time()
print ("training_time", end - start)
#%%
#%%  
#check the predication and loss after training   
predict = sess.run(y_hat, {X: X_train})
print ("loss", sess.run(loss, {X:X_train, y:y_train}))
#
# finding accuracy (with numpy)
train_accuracy = np.mean(np.argmax(y_train, axis=1) == np.argmax(predict, axis=1))
print (train_accuracy)
print ("loss", sess.run(loss, {X:X_train, y:y_train}))
plt.plot(cost)

# finding accuracy with tf's function
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: X_train, y: y_train}))
