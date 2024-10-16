import numpy as np
from read_cifar import *
import matplotlib.pyplot as plt
import os


def distance_matrix(data_train, data_test):
    """ Calculation of the distance matrix """

    train_squared = np.sum(data_train ** 2, axis=1, keepdims=True)
    test_squared = np.sum(data_test ** 2, axis=1, keepdims=True)
    dot_product = np.dot(data_train, data_test.T)
    dists = np.sqrt(train_squared - 2 * dot_product + test_squared.T)

    return dists


def knn_predict(dists,labels_train,k):
    """ To predict the model, for each picture of the test set we find the k-smallest distances and then choose the most recurrent label in those """
    predicted_labels = []

    for i in range(dists.shape[1]):
        k_nearest = np.argpartition(dists[:, i], k)
        labels = labels_train[k_nearest[:k]]
        unique_labels, counts = np.unique(labels, return_counts=True)

        most_frequent_label = unique_labels[np.argmax(counts)]

        predicted_labels.append(most_frequent_label)

    return predicted_labels

def evaluate_knn(data_train,labels_train,data_test,labels_test,k):
    """ To evaluate the model, for each picture of the test set we compare the result of the model with the test result. Then we have the scoring for this model. """
    dists = distance_matrix(data_train,data_test)

    predicted_labels = knn_predict(dists, labels_train, k)

    accuracy = 0
    for i in range(len(labels_test)):
        if predicted_labels[i] == labels_test[i]:
            accuracy += 1

    return accuracy / len(labels_test)


if __name__ == "__main__":

    path = 'data/cifar-10-batches-py/'
    data,labels  = read_cifar_test(path)

    data_train,labels_train,data_test,labels_test = split_dataset(data,labels,0.9)

    accuracy = []
    for k in range(1, 21):
        accuracy.append(evaluate_knn(data_train, labels_train, data_test, labels_test, k))

    # Plot accuracy vs. k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 21), accuracy, marker='o')
    plt.title('KNN Accuracy vs. k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Create directory 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot as knn.png in the results directory
    plt.savefig('results/knn.png')
    plt.show()

