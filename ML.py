import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data loading and preprocessing
data = pd.read_csv('/Users/Your_Name/Desktop/train.csv')  # CHANGE TO YOUR ACCOUNT NAME
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

# Dev data
dev_data = data[0:1000].T
Y_dev = dev_data[0]
X_dev = dev_data[1:n]
X_dev = X_dev / 255

# Training data
training_data = data[1000:m].T
Y_train = training_data[0]
X_train = training_data[1:n]
X_train = X_train / 255
_, m_train = X_train.shape

# Neural network functions
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

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, one_hot_Y):
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
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    history = {'loss': [], 'accuracy': []}
    one_hot_Y = one_hot(Y)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, one_hot_Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 20 == 0:
            loss = -np.mean(np.log(A2[Y, np.arange(m_train)]))
            accuracy = get_accuracy(get_predictions(A2), Y)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            print(f"Iteration: {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return W1, b1, W2, b2, history

# Training the model
W1, b1, W2, b2, history = gradient_descent(X_train, Y_train, 0.10, 501)

# Accuracy on dev dataset
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Dev data accuracy: ", get_accuracy(dev_predictions, Y_dev))

# Testing predictions
test_prediction(0, W1, b1, W2, b2)

# Visualizing Loss and Accuracy
def plot_history(history):
    epochs = range(1, len(history['loss']) + 1) 
    
    plt.figure(figsize=(14, 6))
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

# Confusion Matrix
def plot_confusion_matrix(Y_true, Y_pred, class_names):
    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Getting predictions for the dev set
Y_dev_pred = make_predictions(X_dev, W1, b1, W2, b2)

# Plotting the confusion matrix
class_names = [str(i) for i in range(10)]
plot_confusion_matrix(Y_dev, Y_dev_pred, class_names)
