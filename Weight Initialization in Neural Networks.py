#!/usr/bin/env python
# coding: utf-8

# # Weight Initialization in Neural Networks

# # Initialization
# 
# Training your neural network requires specifying an initial value of the weights. A well chosen initialization method will help learning.  In this notebook, you will see how different initializations lead to different results. 
# 
# A well chosen initialization can:
# - Speed up the convergence of gradient descent
# - Increase the odds of gradient descent converging to a lower training (and generalization) error 
# 
# To get started, run the following cell to load the packages and the planar dataset you will try to classify.

# In[2]:


import torch
from torch import nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import random
random.seed(0)

torch.manual_seed(0)
# Data set 
np.random.seed(1)
train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
np.random.seed(2)
test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
# Visualize the data
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);

# Training and test data
train_x = torch.Tensor(train_X) 
train_y = torch.Tensor(train_Y)

test_x = torch.Tensor(test_X) 
test_y = torch.Tensor(test_Y)


# You would like a classifier to separate the blue dots from the red dots.
# 
# Some helper functions.

# In[3]:


# Some helper functions
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
    y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    d = np.c_[xx.ravel(), yy.ravel()]
    # Predict the function value for the whole grid
    Z = model(torch.from_numpy(d.astype('float32')))
    Z = (Z>.5)*1.
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z.detach().numpy(), cmap=plt.cm.Spectral, alpha=.5)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title('Decision Boundary')
    plt.show()
# plot the results

def plot_cost_history(J_history):
    plt.plot(J_history)
    plt.title('Convergence plot of gradient descent')
    plt.xlabel('No of iterations')
    plt.ylabel('J')
    plt.show()


# ## 1 - Neural Network model 
# You will use a neural network with two hidden layers and an output layer (the model is already implemented for you):
#  - First hidden layer with 10 neuron and ReLU activation function
#  - Second Hidden Layer with 5 neurons and ReLU activation function
#  - Output layer with 1 neuron and sigmoid activation function
#  
# You also need to implement three initialization methods|
# - *init_weights_zero* -- already implemented for you
# - *init_weights_random* -- This initializes the weights to random values with standard deviation equal to 10
# - *He initialization* -- This initializes the weights to random values scaled according to a paper Kaiming He et al., 2015. 

# In[4]:


def init_weights_zero(m):
    with torch.no_grad():
        if type(m) == nn.Linear:
            m.weight.fill_(0.0)
            m.bias.fill_(0.0)
            
class Model(nn.Module):
    def __init__(self, in_feat=2):
        super(Model, self).__init__()
        
        self.feature = nn.Sequential(nn.Linear(in_feat, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 5),
                                   nn.ReLU(),
                                   nn.Linear(5, 1),
                                   nn.Sigmoid()
                                )    
        self.apply(init_weights_zero)    
                
    def forward(self, x):
        return self.feature(x)     
print(Model())

def train(hypothesis, optimizer, Cost, train_x, train_y):
    J_history = []
    for epoch in range(15000):
    
        optimizer.zero_grad()
        # forward pass
        out = hypothesis(train_x)

        loss = Cost(out.flatten(), train_y)

        # backward pass
        loss.backward()

        # update paramters
        optimizer.step()

        if epoch % 1000 == 0:        
            print(f'Iter {epoch+1} Loss: {loss.item()}')

        J_history += [loss.item()]
    return J_history


# ## 2 - Zero initialization
# 
# There are two types of parameters to initialize in a neural network:
# - the weight matrices $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
# - the bias vectors $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$
# 
# Print the weights and carry out the training

# In[5]:


hypothesis = Model()
hypothesis.apply(init_weights_zero)
for n,p in hypothesis.named_parameters():
    print(p)


# In[6]:


optimizer = optim.SGD(hypothesis.parameters(), lr = .01) # stochastic gradient descent with learning rate lr
Cost = nn.BCELoss() # Negative log likelihood loss

J_history = train(hypothesis, optimizer, Cost, train_x, train_y)


# In[7]:


# plot the results
plot_cost_history(J_history)
plot_decision_boundary(hypothesis, train_x, train_y)

out = hypothesis(test_x)
print('Test accuracy: ', torch.sum((out.flatten()>.5)*1 == test_y)/(1.0*test_y.shape[0]))


# The performance is really bad, and the cost does not really decrease, and the algorithm performs no better than random guessing. Why? Lets look at the details of the predictions and the decision boundary:
#     
#     The model is predicting p = 0.5 for every example. 
# 
# In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful than a linear classifier such as logistic regression. 
#     
#     
#     <font color='blue'>
# **What you should remember**:
# - The weights $W^{[l]}$ should be initialized randomly to break symmetry. 
# - It is however okay to initialize the biases $b^{[l]}$ to zeros. Symmetry is still broken so long as $W^{[l]}$ is initialized randomly. 

# ## 3 - Random initialization
# 
# To break symmetry, lets intialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. In this exercise, you will see what happens if the weights are intialized randomly, but to very large values. 
# 
# **Exercise**: Implement the following function to initialize your weights to large random values (scaled by \*10) and your biases to zeros. 

# In[10]:


def init_weights_random(m):
    with torch.no_grad():
        if type(m) == nn.Linear:
            m.weight.fill_(np.random.randn()*10)
            m.bias.fill_(0.0)


# Model would have to be redefined because `self.apply()` was being applied over zero weight initiliazation function

# In[13]:


class Model(nn.Module):
    def __init__(self, in_feat=2):
        super(Model, self).__init__()
        
        self.feature = nn.Sequential(nn.Linear(in_feat, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 5),
                                   nn.ReLU(),
                                   nn.Linear(5, 1),
                                   nn.Sigmoid()
                                )    
        self.apply(init_weights_random)    
                
    def forward(self, x):
        return self.feature(x)   


# In[14]:


hypothesis = Model()
hypothesis.apply(init_weights_random)

optimizer = optim.SGD(hypothesis.parameters(), lr = .01) # stochastic gradient descent with learning rate lr
Cost = nn.BCELoss() # Negative log likelihood loss

J_history = train(hypothesis, optimizer, Cost, train_x, train_y)


# In[15]:


# plot the results
plot_cost_history(J_history)
plot_decision_boundary(hypothesis, train_x, train_y)

out = hypothesis(test_x)
print('Test accuracy: ', torch.sum((out.flatten()>.5)*1 == test_y)/(1.0*test_y.shape[0]))


# **Observations**:
# - The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
# - Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm. 
# - If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.
# 
# <font color='blue'>
# **In summary**:
# - Initializing weights to very large random values does not work well. 
# - Hopefully intializing with small random values does better. The important question is: how small should be these random values be? 

# ## 4 - Kaiming He initialization
# 
# Finally, try "He Initialization"; this is named for the first author of He et al., 2015. 
# 
# **Exercise**: Implement the following function to initialize your parameters with He initialization.

# In[28]:


nn.Linear(123,567).weight


# In[37]:


def init_weights_kaiming_he(m):
    with torch.no_grad():
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)


# In[38]:


class Model(nn.Module):
    def __init__(self, in_feat=2):
        super(Model, self).__init__()
        
        self.feature = nn.Sequential(nn.Linear(in_feat, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 5),
                                   nn.ReLU(),
                                   nn.Linear(5, 1),
                                   nn.Sigmoid()
                                )    
        self.apply(init_weights_kaiming_he)    
                
    def forward(self, x):
        return self.feature(x)   


# In[39]:


hypothesis = Model()
hypothesis.apply(init_weights_kaiming_he)

optimizer = optim.SGD(hypothesis.parameters(), lr = .01) # stochastic gradient descent with learning rate lr
Cost = nn.BCELoss() # Negative log likelihood loss

J_history = train(hypothesis, optimizer, Cost, train_x, train_y)


# In[40]:


# plot the results
plot_cost_history(J_history)
plot_decision_boundary(hypothesis, train_x, train_y)

out = hypothesis(test_x)
print('Test accuracy: ', torch.sum((out.flatten()>.5)*1 == test_y)/(1.0*test_y.shape[0]))


# Observations:
# 
# The model with He initialization separates the blue and the red dots very well.
