# NeuralNetwork
This is a neural built from scratch.
The program's goal is to be able to identify numbers 0-9 even if they are blurry; via this neural network.
Project was based off of Samson Zhang's youtube video building a Neural network from scratch.
<br>
<br>
<br>

## The imports:
```
import numpy as np               # math tool
import pandas as pd              # analysis tool
import matplotlib.pyplot as plt  # plotting tool to show numbers
```
<br>
<br>

## Getting the data read in & set up:
```
# data set is from kaggle https://www.kaggle.com/competitions/digit-recognizer
# set destination to where ever the saved train.csv file is
data = pd.read_csv('/Users/Your_Name/Desktop/train.csv')
# optional print
# print(data.head())
# change to numpy
data = np.array(data)
# optional print
print(data)
```
<br>
<br>

## Declaration and initialization of variables:
```
# m = rows, n = columns + 1 (for label column)
m,n = data.shape
# shuffle before splitting into dev and training sets
np.random.shuffle(data)

# dev_data is the first 1000 examples
# .T to transpose (from rows to columns)
dev_data = data[0:1000].T
Y_dev = dev_data[0]
# 1 : max columns
X_dev = dev_data[1:n]
X_dev = X_dev / 255

# training data (rest of data)
training_data = data[1000:m].T
Y_train = training_data[0]
X_train = training_data[1:n]
X_train = X_train / 255
_,m_train = X_train.shape
```
<br>

### What is in the data set?
Images are represented digitally with pixel values ranging from 0 to 255, indicating the intensity of light or color. This is what the X data contains.  
The Y data contains the actual number the image has (0-9).
<br>

### Why do we divide the X data sets by 255?
When you divide all pixel values by 255 in the dataset ( what X data has ), you're performing a normalization step where all features (pixel values, in this case) will have a similar range of values, specifically between 0 and 1. It helps speed up gradient descent, reduces the chance of getting stuck in local optima, and that a feature of a number doesn't have a disproportionate importance.
<br> 
<br>
<br>

## Now we train the data:

w1: Weights connecting the input layer ( 784 units is designed for something like the MNIST dataset of 28x28 pixel images ) to the hidden layer (with 10 neurons). <br>
    
b1: Biases for each of the 10 neurons in the hidden layer.<br>

w2: Weights for connections between the hidden layer's 10 neurons and the 10 output neurons. <br>

b2: Biases for the 10 neurons in the output layer. <br>
<br>

**Initializes the parameters for a simple neural network with one hidden layer. 
It creates random weights (w1, w2) and biases (b1, b2) for both the input-to-hidden and hidden-to-output layer connections.**
```
def init_params():
    # Input layer size (28x28 image)
    n_in = 784
    # Size of the first layer (number of neurons in the first hidden layer)
    n_hidden1 = 10
    # Size of the second layer (number of neurons in the output layer)
    n_hidden2 = 10

    # Initialize weights and biases for the first layer
    # w1 is a n_hidden1 x n_in matrix, the . after the 2 is for floating point division
    W1 = np.random.randn(n_hidden1, n_in) * np.sqrt(2. / (n_in + n_hidden1))
    b1 = np.zeros((n_hidden1, 1))  # Biases can be initialized to zeros

    # Initialize weights and biases for the second layer
    # w1 is a n_hidden2 x n_hidden1 matrix, the . after the 2 is for floating point division
    W2 = np.random.randn(n_hidden2, n_hidden1) * np.sqrt(2. / (n_hidden1 + n_hidden2))
    b2 = np.zeros((n_hidden2, 1))  # Biases can be initialized to zeros
    return W1, b1, W2, b2
```
<br>

### What initialization is used?
He initialization is used.  It helps avoid diminishing or exploding gradients during training by ensuring that the variance of the outputs of each layer remains controlled, thus making the network more likely to learn effectively.
<br>
<br>
<br>

## Rectified Linear Unit ( RelU )
**Goes through given matrix Z and returns the max when compared to 0,
effectively replaces numbers <0 with 0.**
```
def ReLU(Z):
    return np.maximum(Z, 0)
```
<br>

**Calculates the derivative of the Rectified Linear Unit (ReLU) activation function with respect to its input Z. 
It returns a Boolean array that is True for elements where Z is greater than 0 and False otherwise. 
This can be multiplied, specifically by numbers.  True * num = num, False * num = 0.**
```
def ReLU_derivative(Z):
    return Z > 0
```
<br>
<br>

## Softmax:
**The softmax function converts a vector of raw scores (often called logits) from the final layer of a neural network 
into probabilities by performing two main steps: <br> <br>
    Exponentiation: It takes the exponential (e raised to the power) of each score. <br>
    This step ensures that all output values are non-negative. Since exponential functions grow rapidly, 
    larger scores become significantly larger than smaller or negative scores, amplifying differences between them. <br> <br>
    Normalization: It then divides each exponentiated score by the sum of all exponentiated scores in the vector. 
    This step ensures that all the output values are between 0 and 1 and that their sum equals 1.**
```
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
```
<br>

z1: Linear Combination for Hidden Layer: Computes z1 as the dot product of w1 and X, then adds b1. 
This results in the pre-activation values for the hidden layer.

a1: ReLU Activation: Applies the ReLU function to z1 to get a1, the activated values for the hidden layer. 
ReLU introduces non-linearity, turning negative values to 0 while keeping positive values unchanged.

z2: Linear Combination for Output Layer: Computes z2 using the activated values a1, the weights w2, and biases b2. 
This gives the pre-activation values for the output layer.

a2: Softmax Activation: Applies the softmax function to z2, resulting in a2, which are the output probabilities of the network. 
Softmax ensures these probabilities sum up to 1, making the output suitable for classification tasks.
<br>
<br>
<br>

## Forward Propagation
```
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2
```
<br>
<br>

## One Hot
Converts a 1D array of integer labels (Y) into a 2D one-hot encoded matrix, 
where each row corresponds to a class and each column to a sample, with 1 indicating the presence of a class for a sample.

```
def one_hot(Y):
    # Creates 2d array of 0's
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
```
np.arange(Y.size): This creates an array of indices for each element in Y. For example, if Y has 5 elements, np.arange(Y.size) produces [0, 1, 2, 3, 4]. 
These represent the row indices in the one_hot_Y matrix where we want to set values.

Y: This is the array of labels you want to one-hot encode. Each value in Y specifies the column index in the one_hot_Y 
matrix where a 1 should be placed for the corresponding row.
    
one_hot_Y[np.arange(Y.size), Y] = 1: This operation simultaneously selects a row and a column in the one_hot_Y matrix 
and sets that position to 1. It effectively goes through each element in Y, uses the element's value as a column index, 
and the element's position in Y as a row index, then places a 1 in the one_hot_Y matrix at that row and column.
    ```
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
    ```


```
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 1)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 1)
    return dW1, db1, dW2, db2
```

```
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))
    return W1, b1, W2, b2
```

```
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```

```
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```

```
def get_predictions(A2):
    return np.argmax(A2, 0)
```

```
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
```

```
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```



W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


```
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)
test_prediction(7, W1, b1, W2, b2)
test_prediction(8, W1, b1, W2, b2)
test_prediction(9, W1, b1, W2, b2)
```
