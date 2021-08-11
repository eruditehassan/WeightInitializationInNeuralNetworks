# Weight Initialization In Neural Networks
Using techniques like Zero Initialization, Random Initialization and Kaiming He initialization for initializing weights for Neural Networks

Training neural network requires specifying an initial value of the weights. A well chosen initialization method will help learning. 
A well chosen initialization can:

- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error


Following techniques were used for Weight Initialization

1. `init_weights_zero` -- This initializes weights with zero
2. `init_weights_random` -- This initializes the weights to random values with standard deviation equal to 10
3. `He initialization` -- This initializes the weights to random values scaled according to a paper Kaiming He et al., 2015.

