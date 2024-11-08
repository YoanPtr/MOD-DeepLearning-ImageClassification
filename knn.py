import numpy as np
from read_cifar import *
import matplotlib.pyplot as plt
import os


def distance_matrix(M1, M2) :
    M1_2 = np.sum(M1**2, axis = 1, keepdims = True)
    M2_2 = np.sum(M2**2, axis = 1, keepdims = True)
    M1M2 = np.dot(M1, M2.T)
    dists = np.sqrt(M1_2 + M2_2.T - 2*M1M2)
    return dists




def knn_predict(dists, labels_train, k):
    """Predicts labels based on k-nearest neighbors algorithm."""
    predicted_labels = []

    for i in range(dists.shape[0]):
        # Use argsort to get the indices of sorted distances in ascending order
        sorted_indices = np.argpartition(dists[i,:], range(k))[:k]
        labels = labels_train[sorted_indices]

        # Find the most frequent label among the k-nearest neighbors
        unique_labels, counts = np.unique(labels, return_counts=True)
        most_frequent_label = unique_labels[np.argmax(counts)]

        predicted_labels.append(most_frequent_label)

    return predicted_labels


def evaluate_knn(dists, labels_train, labels_test, k):
    """Evaluates the KNN model accuracy by comparing predictions to true labels."""
    predicted_labels = knn_predict(dists, labels_train, k)

    # Efficient calculation of accuracy
    accuracy = np.mean(np.array(predicted_labels) == labels_test)

    return accuracy


if __name__ == "__main__":
    path = 'data/cifar-10-batches-py/'
    data, labels = read_cifar_test(path)

    data = normalized(data)
    
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.9)

    # Compute distance matrix once for efficiency
    dists = distance_matrix(data_test, data_train)
    accuracy = []

    # Evaluate for k values from 1 to 20
    for k in range(1, 21):
        accuracy.append(evaluate_knn(dists, labels_train, labels_test, k))
        print(f"Accuracy for k={k}: {accuracy[-1]:.2f}")

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
