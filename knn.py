import numpy as np
from read_cifar import *
import matplotlib.pyplot as plt
import os


def distance_matrix(M1, M2):
    """Computes the Euclidean distance matrix between two sets of vectors."""
    expanded_M1 = np.expand_dims(M1, axis=1)
    expanded_M2 = np.expand_dims(M2, axis=0)

    squared_diff = np.sum((expanded_M1 - expanded_M2) ** 2, axis=2)
    dists = np.sqrt(squared_diff)

    return dists


def knn_predict(dists, labels_train, k):
    """Predicts labels based on k-nearest neighbors algorithm."""
    predicted_labels = []

    for i in range(dists.shape[1]):
        # Use argsort to get the indices of sorted distances in ascending order
        sorted_indices = np.argsort(dists[i, :])[:k]
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

    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.9)

    # Compute distance matrix once for efficiency
    dists = distance_matrix(data_train, data_test)
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
