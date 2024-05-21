import numpy as np               # math tool
import pandas as pd              # analysis tool
import matplotlib.pyplot as plt  # plotting tool to show numbers

# data set is from kaggle https://www.kaggle.com/competitions/digit-recognizer
# set destination to where ever the saved train.csv file is
data = pd.read_csv('/Users/Over Yonder/Desktop/train.csv')
data = np.array(data)
# optional print
print(data)

# m = rows, n = columns + 1 (for label column)
m,n = data.shape
np.random.shuffle(data)

# dev_data is the first 1000 examples
# .T to transpose (from rows to columns)
dev_data = data[0:1000].T
Y_dev = dev_data[0]
X_dev = dev_data[1:n]
X_dev = X_dev / 255

# training data (rest of data)
training_data = data[1000:m].T
Y_train = training_data[0]
X_train = training_data[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

def init_params():
    n_in = 784
    n_hidden1 = 30
    output = 10

    W1 = np.random.randn(n_hidden1, n_in) * np.sqrt(2. / n_in)
    b1 = np.zeros((n_hidden1, 1)) 

    W2 = np.random.randn(output, n_hidden1) * np.sqrt(2. / n_hidden1)
    b2 = np.zeros((output, 1))  

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_derivative(Z):
    return Z > 0

def softmax(Z):
    Z -= np.max(Z, axis=0) 
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def one_hot(Y):
    # Creates 2d array of 0's
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 1)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (30, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def get_predictions(A2):
    return np.argmax(A2, 0)
    
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

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # print every 20th iteration
        if i % 20 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("%.3f" % get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# shows predictions, add more if wanted 
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
