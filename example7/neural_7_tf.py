#  solving linear regression example "https://www.tensorflow.org/get_started/get_started"
# This code just for testing tensorflow 
import tensorflow as tf 
#%%
import os 
# if do not use the below line, by default it uses gpu, and tesla4 is not 
#compataible driver to run with gpu for tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"]=""
import pandas as pd 
#%%
# getting data and normalization
inputs = [1,2,3,4]
outputs = [0,-1,-2,-3]
#%%
# hyper parameters
# placeholders
X = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
# variables 
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
#%%
# forward propogation 
y_hat = W * X + b 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
print (sess.run(init))
print(sess.run(y_hat, {X: inputs }))
#%%
# loss function and optimizer
cost = tf.reduce_sum(tf.square(y_hat - y))
print (sess.run(cost, {X: inputs, y: outputs}))
#%%
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)
#%%
# training
for i in range(1000):
    sess.run(train, {X: inputs,y: outputs})
    
curr_W, curr_b, curr_loss = sess.run([W, b, cost], {X: inputs, y: outputs})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
