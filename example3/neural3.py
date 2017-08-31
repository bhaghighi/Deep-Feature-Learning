#example: https://github.com/dennybritz/nn-theano/blob/master/nn-theano.ipynb
import theano 
import theano.tensor as T
#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
#%%
np.random.seed(0)
train_X, train_y = sklearn.datasets.make_moons(200, noise=0.20)
train_X = train_X.astype("float32")
train_y = train_y.astype("int32")
print (train_X.shape, train_y.shape)
plt.scatter(train_X[:,0],train_X[:,1],c= train_y)
#%%
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = train_X[:, 0].min() - .5, train_X[:, 0].max() + .5
    y_min, y_max = train_X[:, 1].min() - .5, train_X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)
#%%
#deinging hyper parameters
num_examples = len(train_X) 
nn_input_dim = 2 
nn_output_dim = 2 
nn_hdim = 100 
epsilon = 0.01 
reg_lambda = 0.01 
#%%
X = T.dmatrix("X")
y = T.lvector("y")
W1 = theano.shared(np.random.randn(nn_input_dim,nn_hdim), name = "W1")
b1 = theano.shared(np.zeros(nn_hdim),name = "b1")
W2 = theano.shared(np.random.randn(nn_hdim,nn_output_dim), name = "W2")
b2 = theano.shared(np.zeros(nn_output_dim), name = "b2")
#%%
# forward propogation 
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)
forward = theano.function([X],y_hat)
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() 
calculate_loss = theano.function([X, y], loss)
prediction = T.argmax(y_hat, axis=1)
predict = theano.function([X],prediction)
#%%
# Backpropogation 
dW1,db1,dW2,db2 = T.grad(loss, [W1,b1,W2,b2])
gradient_step = theano.function(
    [X, y],
    updates=((W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)))

#%%
def build_model(num_passes=20000, print_loss=False):
    
    # Re-Initialize the parameters to random values. We need to learn these.
    # (Needed in case we call this function multiple times)
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_output_dim))
    
    # Gradient descent. For each batch...
    for i in range(num_passes):
        # This will update our parameters W2, b2, W1 and b1!
        gradient_step(train_X, train_y)
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print ("Loss after iteration %i: %f" %(i, calculate_loss(train_X, train_y)))
            
#%%
import time 
start = time.time()
build_model(print_loss=True)
end = time.time()
print (end - start)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(x))
plt.title("Decision Boundary for hidden layer size 3")














