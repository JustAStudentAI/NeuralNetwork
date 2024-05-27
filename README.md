# NeuralNetwork
This is a neural built from scratch, the purpose is to identify numbers 0-9 from a given image.  I used the MNIST dataset for this project, referenced as "train.csv". 
Project was based off of Samson Zhang's youtube video building a Neural network from scratch. https://www.youtube.com/watch?v=w8yWXqWQYmU
<br>
<br>
<p align="center">
  <img width="697" alt="Neural net" src="https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/04a0ea34-7b62-4404-ac9e-796cd56be081">
</p>


<br>
<br>

## Model Architecture

The neural network is designed with the following architecture:
- **Input Layer:** 784 neurons, representing a 28x28 pixel image with a number 0-9 in it.
- **Hidden Layer:** A single hidden layer with 30 neurons, utilizing ReLU activation functions to introduce non-linearity.
- **Output Layer:** 10 neurons, representing 0-9 using a softmax activation function for multi-class classification.

This configuration allows the model to learn complex patterns efficiently, tailored to the specific demands of the dataset.
<br>
<br>
<br>

## The imports:
To download, type this into terminal: pip3 install numpy pandas scikit-learn seaborn

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
```
<br>

### What is each import used for?
to be finished~~~~~~~~~~
<br>
<br>

## Getting the data read in & set up:
Data set is from kaggle https://www.kaggle.com/competitions/digit-recognizer
```
# set destination to where ever the saved train.csv file is
data = pd.read_csv('/Users/Your_Name/Desktop/train.csv')
# change to numpy
data = np.array(data)
# optional print
# print(data)
```
<br>

### What is MNIST ( the dataset )?
The MNIST database ( Modified National Institute of Standards and Technology database ) is a large database of handwritten digits that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images, 
and is also widely used for training and testing in the field of machine learning.  ( wikipedia )
<br>
<br>
<p align="center">
    <img width="700" alt="data set" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/f8a922d7-b548-44d7-9da7-3a22f517c081> <br>
      <i>
      ( Wikipedia )
      </i>
</p>
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

### Why do we divide the X data sets by 255?
Images are represented digitally with pixel values ranging from 0 to 255, indicating the intensity of light or color. This is what the X data contains.  
The Y data contains the actual number the image has (0-9). When you divide all pixel values by 255 in the dataset ( what X data has ), you're performing a normalization step where all features (pixel values, in this case) will have a similar range of values, specifically between 0 and 1. 
It helps speed up gradient descent, reduces the chance of getting stuck in local optima, and that a feature of a number doesn't have a disproportionate importance.
<br> 
<br>
<br>

## Now we train the data:

w1: Weights connecting the input layer ( 784 units is designed for something like the MNIST dataset of 28x28 pixel images ) to the hidden layer (with 10 neurons). <br>
    
b1: Biases for each of the 10 neurons in the hidden layer.<br>

w2: Weights for connections between the hidden layer's 10 neurons and the 10 output neurons. <br>

b2: Biases for the 10 neurons in the output layer. <br>
<br>

Initializes the parameters for a simple neural network with one hidden layer. 
It creates random weights (w1, w2) and biases (b1, b2) for both the input-to-hidden and hidden-to-output layer connections.
```
def init_params():
    # Input layer size (28x28 image)
    n_in = 784
    # Size of the first layer (number of neurons in the first hidden layer)
    n_hidden1 = 30
    # Size of the output layer
    output = 10

    # Initialize weights and biases for the first layer
    # He initialization: only consider fan-in (n_in)
    W1 = np.random.randn(n_hidden1, n_in) * np.sqrt(2. / n_in)
    b1 = np.zeros((n_hidden1, 1))  # Biases can be initialized to zeros

    # Initialize weights and biases for the second layer
    # He initialization: only consider fan-in (n_hidden1)
    W2 = np.random.randn(output, n_hidden1) * np.sqrt(2. / n_hidden1)
    b2 = np.zeros((output, 1))  # Biases can be initialized to zeros

    return W1, b1, W2, b2
```
<br>

### What initialization is used?
He initialization is used.  It helps avoid diminishing or exploding gradients during training by ensuring that the variance of the outputs of each layer remains controlled, thus making the network more likely to learn effectively.  
Note: This was a random choice to use HE; multiplying by .05 instead would result in a ~2% loss of accuracy compared to HE initialization.

More information: https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
<br>
<br>
<p align="center">
  <img width="318" alt="HE ini" src="https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/8b7884ad-c553-4883-ad82-28dbe02ca92c"> <br>
  <i>
    ( Naver )
  </i>
</p>
<br>
<br>

## Rectified Linear Unit ( RelU )
Goes through given matrix Z and returns the max when compared to 0,
effectively replaces numbers <0 with 0.**
```
def ReLU(Z):
    return np.maximum(Z, 0)
```
<br>
<p align="center">
  <img width="397" alt="relU" src="https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/2ee4aa11-826b-4b3c-938c-691bebb30977"> <br>
  <i>
    ( LinkedIn )
  </i>
</p>
<br>

Calculates the derivative of the Rectified Linear Unit (ReLU) activation function with respect to its input Z. 
It returns a Boolean array that is True for elements where Z is greater than 0 and False otherwise. 
This can be multiplied, specifically by numbers.  True * num = num, False * num = 0.
```
def ReLU_derivative(Z):
    return Z > 0
```
<br>
<br>

## Softmax:
The softmax function converts a vector of raw scores (often called logits) from the final layer of a neural network 
into probabilities by performing two main steps: <br> <br>
    Exponentiation: It takes the exponential (e raised to the power) of each score. <br>
    This step ensures that all output values are non-negative. Since exponential functions grow rapidly, 
    larger scores become significantly larger than smaller or negative scores, amplifying differences between them. <br> <br>
    Normalization: It then divides each exponentiated score by the sum of all exponentiated scores in the vector. 
    This step ensures that all the output values are between 0 and 1 and that their sum equals 1.
```
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
```
<p align="center">
 <img width="500" alt="softmax" src="https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/126d2562-9d82-436f-bc79-6c388f45c9bb">
</p>
<br>
<br>

## Forward Propagation
Input data is passed through a neural network, layer by layer, to generate output. This output is then used to calculate the error of the model during training.

### What are the parameters?
w1: Weight 1 <br>
b1: Bias 2  <br>
w2: Weight 2  <br>
b2: Bias 2  <br>
X: Input   <br>

### What are the variables in the function?
z1: Linear Combination for Hidden Layer: Computes z1 as the dot product of w1 and X, then adds b1. 
This results in the pre-activation values for the hidden layer. <br>

a1: ReLU Activation: Applies the ReLU function to z1 to get a1, the activated values for the hidden layer. 
ReLU introduces non-linearity, turning negative values to 0 while keeping positive values unchanged.

z2: Linear Combination for Output Layer: Computes z2 using the activated values a1, the weights w2, and biases b2. 
This gives the pre-activation values for the output layer.

a2: Softmax Activation: Applies the softmax function to z2, resulting in a2, which are the output probabilities of the network. 
Softmax ensures these probabilities sum up to 1, making the output suitable for classification tasks.
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


## Backward Propagation
Calculates the error from the output and distributes it back through the network layers. This helps in adjusting the model's parameters to reduce errors and improve accuracy during learning.
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
<br>
<br>

## One Hot
Converts a 1D array of integer labels (Y) into a 2D one-hot encoded matrix, 
where each row corresponds to a class and each column to a sample, with 1 indicating the presence of a class for a sample.
```
def one_hot(Y):
    # Creates 2d array of 0's
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```
<p align="center">
 <img width="500" alt="one hot" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/b408530e-c935-4336-b417-aafe3ae717b2>
</p>
<br>
<br>

## Update parameters
```
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (30, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))
    return W1, b1, W2, b2
```

### What is alpha?
Alpha, also known as the learning rate, is a hyperparameter that determines the step size at each iteration of the gradient descent algorithm.
<br>
<br>
<br>

## Predictions
The numpy.argmax() function returns indices of the max element of the array in a particular axis. This results in the return value being 0-9.
```
def get_predictions(A2):
    return np.argmax(A2, 0)
```


Gets the prediction for given parameters, does this by inputting it into forward prop and using get_predictions to get the prediction.
```
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```

Prints out the prediction, actual number, as well as the image used from the MNIST dataset.
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
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
```
<br>
<br>

## Gradient descent
### What is gradient descent?
Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. 
Gradient descent in machine learning is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.
<p align="center">
 <img width="600" alt="grad des" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/7b9d385d-98c9-4016-b58f-5b8024c36476>
</p>
<br>

```
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    history = {'loss': [], 'accuracy': []}
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # print some data every 20th iteration
        if i % 20 == 0:
            loss = -np.mean(np.log(A2[Y, np.arange(m_train)]))
            accuracy = get_accuracy(get_predictions(A2), Y)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            print(f"Iteration: {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return W1, b1, W2, b2, history
```
<p align="center">
 <img width="600" alt="iteration ss" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/06730932-52bc-49b6-865c-ac485471f337>
</p>
<br>

### What is a loss / cost function?
A loss / cost function is used to measure just how wrong the model is in finding a relation between the input and output. It tells you how badly your model is behaving/predicting
<p align="center">
 <img width="500" alt="cost func" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/d43e2c11-cb76-49d9-b493-5a6b259afc05>
</p>
<br>

More information: <br>
https://builtin.com/data-science/gradient-descent <br> 
https://www.simplilearn.com/tutorials/machine-learning-tutorial/cost-function-in-machine-learning <br>
<br>

### Training the model
Calls gradient descent, passes the data sets with alpha = 0.10 and 501 iterations.
```
W1, b1, W2, b2, history = gradient_descent(X_train, Y_train, 0.10, 501)
```
<br>
<br>

## Show prediction visuals
```
# shows predictions, add more if wanted 
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
```
<p align="center">
 <img width="400" alt="ss" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/1c9a3a56-e512-4966-9939-ebe1ff3c0e1a>
</p>
<p align="center">
 <img width="400" alt="ss" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/b5dc9e37-b1ec-42d2-ac8b-00f305a0d347>
</p>
<br>
<br>

## Visualize loss and accuracy
This creates two subplots. The first subplot shows the training loss versus epochs, and the second subplot shows the training accuracy versus epochs. This function helps in understanding how the model's performance evolves during training by displaying the trends in loss and accuracy.
```
def plot_history(history):
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

plot_history(history)
```
<p align="center">
 <img width="1000" alt="ss" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/de7561ed-0cc2-4f5c-8e03-08c7fda7c741>
</p>
<br>
<br>

## Confusion Matrix
The confusion matrix compares the true labels to the predicted labels, showing the number of correct and incorrect predictions for each class. This function helps in understanding the model's performance across different classes and identifying any patterns of misclassification.
```
def plot_confusion_matrix(Y_true, Y_pred, class_names):
    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

Y_dev_pred = make_predictions(X_dev, W1, b1, W2, b2)

class_names = [str(i) for i in range(10)]
plot_confusion_matrix(Y_dev, Y_dev_pred, class_names)
```
<p align="center">
 <img width="800" alt="ss" src=https://github.com/JustAStudentAI/NeuralNetwork/assets/132246011/76b22eb3-8050-427e-9d5f-eabab4fcde5e>
</p>
<br>
<br>

## References
Bourke, D. (2023, September). How to use non-linear functions in neural networks. LinkedIn. https://www.linkedin.com/posts/mrdbourke_machinelearning-datascience-neuralnetworks-activity-7107129007233515520-3rpF <br>
Duif, M. (2020, January 10). Exploring How Neural Networks Work and Making Them Interactive. Medium. https://towardsdatascience.com/exploring-how-neural-networks-work-and-making-them-interactive-ed67adbf9283 <br>
Wikipedia Contributors. (2019, February 22). MNIST database. Wikipedia; Wikimedia Foundation. https://en.wikipedia.org/wiki/MNIST_database <br>
Zinc. (2019, June 19). [Summary] [PyTorch] Lab-09-2 Weight initialization. Blog.naver.com. https://blog.naver.com/hongjg3229/221564537122 <br>



