# MNIST with tensorflow (one hidden layer, a question? i added a hidden layer
# but i could not get the good accuracy like no hidden layer? WHY?)
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
# see the four first images 
plt.subplot(221)
plt.imshow(X_train[0].reshape(28,28), cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1].reshape(28,28), cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2].reshape(28,28), cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3].reshape(28,28), cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
#%%
# define hyperparmaters of the network
inputLayerSize = 784 # 28 * 28
hiddenLayerSize = 100
outputLayerSize = 10
#
X = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
W1 = tf.Variable(tf.zeros([inputLayerSize,hiddenLayerSize]), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([hiddenLayerSize]), dtype=tf.float32)
W2 = tf.Variable(tf.zeros([hiddenLayerSize,outputLayerSize]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([outputLayerSize]), dtype=tf.float32)
#%%
# forward propogation 
z1 = tf.matmul(X, W1) +  b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1,W2) + b2
y_hat = z2 # do not apply softmax if you use logit cross entropy as 
# it applies softmax internally
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
print (sess.run(y_hat,{X: X_train}))
#%%
# defining cost functiona and optimizer 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
cost = []
start = time.time()
for i in range(100):
    sess.run(train, feed_dict={X:X_train,y:y_train})
    cost.append(sess.run(loss,feed_dict={X:X_train,y:y_train}))
end = time.time()
print ("training_time", end - start)
#%%  
#check the predication and loss after training   
predict = sess.run(y_hat, {X: X_train})
print ("loss", sess.run(loss, {X:X_train, y:y_train}))
#
# finding accuracy
train_accuracy = np.mean(np.argmax(y_train, axis=1) == np.argmax(predict, axis=1))
print (train_accuracy)
print ("loss", sess.run(loss, {X:X_train, y:y_train}))
plt.plot(cost)

# 
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: X_train, y: y_train}))






