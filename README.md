# Basic Neural Network
---
- Input layer (receives data and passes it on)
- Hidden layer
- Output layer
- Weights between the layers
- Activation function (per hidden layer) (Sigmoid activation function)
- Feed-forward (perception) neural network (relays data directly from the front to the back)


 ![](https://www.kdnuggets.com/wp-content/uploads/simple-neural-network.png)

`input * weight + bias = output`


### Training Data
|          | Input       |  Output |
|:---------|:-----------:|:-------:|
| Data 1   |  0 - 0 - 1  |    0    |
| Data 2   |  1 - 1 - 1  |    1    |
| Data 3   |  1 - 0 - 1  |    1    |
|          |             |         |
| New Case |  1 - 0 - 0  |    ?    |

Every input will have a weight (positive or negative).
This implies that an input having a big number of positive weight or a big number of negative weight will influence the resulting output more procedure for the training process
taking inputs from the training dataset, performing some adjustments based on their weights, and siphoning them via a method that computes the output of the neural net
compute the back-propogated error rate. in this case it is the difference between neuron's predicted output and the expected output of the training dataset
based on the extent of the error, perform some minor weight adjustments using the Error Weighted Derivative formula
iterate this process an arbitrary number of 15000 times, every iteration the whole training set is processed simultaneously

- `exp`    - for generating the natural exponential
- `array`  - for generating a matrix
- `dot`    - for multiplying matrices
- `random` - for generating random numbers (we'll seed the random numbers to ensure their efficient distribution)
