import numpy as np
from read_cifar import *
import matplotlib.pyplot as plt
import os
import random

def sigmoid(x):
    """Calculation of the sigmoid function for a numpy array"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Calculation of the sigmoid derivative function for a numpy array """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """
    Compute the softmax of each row of the input x.
    Each row represents a set of scores, and softmax normalizes them into probabilities.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def learn_once_mse(w1,b1,w2,b2,data,targets,learning_rate):
    """
    Perform one gradient descent step using MSE loss
    """
    a0 = data  # Input to the first layer
    z1 = np.matmul(a0, w1) + b1  # Input to the hidden layer
    a1 = sigmoid(z1)  # Output of the hidden layer
    z2 = np.matmul(a1, w2) + b2  # Input to the output layer
    a2 = sigmoid(z2)  # Output of the output layer
    y_pred = a2  # Predictions

    loss = np.mean(np.square(y_pred - targets))

    # Gradient of the loss
    d_loss_a2 = 2 / len(targets) * ( y_pred - targets)
    d_loss_z2 = np.matmul(d_loss_a2 , sigmoid_derivative(a2))

    # Layer 2
    d_loss_w2 = np.matmul(a1.T,d_loss_z2 ) / data.shape[0]
    d_loss_b2 = np.mean(d_loss_z2, axis=0)

    # Layer 1

    d_loss_a1 = np.matmul(d_loss_z2 , w2.T)
    d_loss_z1 = d_loss_a1 * a1 * (1 - a1)

    d_loss_w1 = np.matmul(a0.T,d_loss_z1 ) / data.shape[0]
    d_loss_b1 = np.mean(d_loss_z1, axis=0)


    # Update weights and biases
    w1 -= learning_rate * d_loss_w1
    b1 -= learning_rate * d_loss_b1
    w2 -= learning_rate * d_loss_w2
    b2 -= learning_rate * d_loss_b2

    return w1,b1,w2,b2,loss

def one_hot(labels,y_pred):
    """
    Perform one hot encoding for the n labels of the dataset
    """
    one_hot_array = np.zeros(y_pred)

    for i in range(len(labels)):
        one_hot_array[i,labels[i]] = 1

    return  one_hot_array


def cross_entropy(y, y_pred):
    """
    Function that calculates the cross entropy between predicted
    and target values
    """
    epsilon = 10 ** (-2)
    return -np.sum(y * np.log(y_pred + epsilon)) / float(y_pred.shape[0])


def learn_once_cross_entropy(w1,b1,w2,b2,data,labels_train,learning_rate):

    """
    Perform one gradient descent step using cross entropy loss
    """

    a0 = data  # Input to the first layer
    z1 = np.matmul(a0, w1) + b1  # Input to the hidden layer
    a1 = sigmoid(z1)  # Output of the hidden layer
    z2 = np.matmul(a1, w2) + b2  # Input to the output layer
    a2 = softmax(z2)  # Output of the output layer
    y_pred = a2  # Predictions

    y = one_hot(labels_train,y_pred.shape)

    loss = cross_entropy(y_pred, y)
    d_loss_z2 = a2 - y

    # Layer 2
    d_loss_w2 = np.matmul(a1.T,d_loss_z2 ) / data.shape[0]
    d_loss_b2 = np.mean(d_loss_z2, axis=0)

    # Layer 1

    d_loss_a1 = np.matmul(d_loss_z2 , w2.T)
    d_loss_z1 = d_loss_a1 * a1 * (1 - a1)

    d_loss_w1 = np.matmul(a0.T,d_loss_z1 ) / data.shape[0]
    d_loss_b1 = np.mean(d_loss_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * d_loss_w1
    b1 -= learning_rate * d_loss_b1
    w2 -= learning_rate * d_loss_w2
    b2 -= learning_rate * d_loss_b2

    return w1, b1, w2, b2, loss

def train_mlp(w1, b1, w2, b2, data, labels_train, learning_rate, num_epoch):
    """
    Run num_epoch batches of gradient descent step using cross entropy loss to train the model
    Return train weights and accuracy of the model on training data
    """

    train_accuracies = []

    for i in range(num_epoch):
        # Perform one step of learning with cross-entropy loss
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate)

        # Forward pass to compute predictions
        a0 = data # Input to the first layer
        z1 = np.matmul(a0, w1) + b1  # Input to the hidden layer
        a1 = sigmoid(z1)  # Output of the hidden layer
        z2 = np.matmul(a1, w2) + b2  # Input to the output layer
        a2 = sigmoid(z2)  # Output of the output layer
        y_pred = a2  # Predictions

        # Compute accuracy
        predict_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(labels_train == predict_classes)
        train_accuracies.append(accuracy)

        # Print progress
        print(f"Epoch {i + 1}/{num_epoch}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

    return w1, b1, w2, b2, train_accuracies


def run_test_mlp(w1,b1,w2,b2,data_test,labels_test):
    """
    Return accuracy of the train model on test data
    """

    a0 = data_test  # Input to the first layer
    z1 = np.matmul(a0, w1) + b1  # Input to the hidden layer
    a1 = sigmoid(z1)  # Output of the hidden layer
    z2 = np.matmul(a1, w2) + b2  # Input to the output layer
    a2 = sigmoid(z2)  # Output of the output layer
    y_pred = a2  # Predictions

    # Compute accuracy
    predict_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(labels_test == predict_classes)

    return accuracy,predict_classes

def run_mlp_training(data_train,labels_train,data_test,labels_test,d_h,learning_rate,num_epoch):


    d_in = data_train.shape[1]  # input dimension
    d_out = labels_train.max() + 1   # output dimension (number of neurons of the output layer)

    # Random initialization of the network weights and biaises
    #w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    #w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights

    w1 = np.random.randn(d_in, d_h) * np.sqrt(1 / d_in)  # Initialisation plus petite
    w2 = np.random.randn(d_h, d_out) * np.sqrt(1 / d_h)

    b1 = np.zeros((1, d_h))  # first layer biaises
    b2 = np.zeros((1, d_out))  # second layer biaises

    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)

    accuracy,predict_classes = run_test_mlp(w1,b1,w2,b2,data_test,labels_test)

    return(train_accuracies,accuracy,predict_classes)

def save_figure(data,labels,label_names,predict_classes):
    plt.figure(figsize=(10, 10))

    # Select 9 random indices
    random_indices = random.sample(range(len(data)), 9)

    # Loop through each random index and plot the image
    for i, idx in enumerate(random_indices):
        # Reshape the image data from 3072 to 32x32x3
        image = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)

        # Add a subplot in a 3x3 grid
        plt.subplot(3, 3, i + 1)

        # Display the image
        plt.imshow(image)

        # Set the title as the label
        plt.title(
            f'Label Train: {label_names[labels[idx]]}\nPredicted Label: {label_names[predict_classes[idx]]}',
            fontsize=10,  # Reduce font size
            pad=10        # Add padding
        )
        # Turn off the axis for cleaner visualization
        plt.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig('results/random_images.png')

    # Show the plot
    plt.show()

if __name__ == "__main__":

    data = 'random'

    path = 'data/cifar-10-batches-py/'
    data,labels  = read_cifar(path)
    label_names = get_label(path)

    data = normalized(data)
    split_factor = 0.9
    d_h = 64
    learning_rate = 0.1
    num_epoch = 100

    data_train,labels_train,data_test,labels_test = split_dataset(data,labels,0.9)

    train_accuracies,accuracy,predict_classes = run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch)
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Test Accuracy: {accuracy:.2f}')


    plt.plot(range(1, num_epoch + 1 ), train_accuracies, marker='o')
    plt.title('MLP Accuracy vs. num_epoch')
    plt.xlabel('num_epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Create directory 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot as knn.png in the results directory
    plt.savefig('results/mlp.png')

    save_figure(data_test,labels_test,label_names,predict_classes)
