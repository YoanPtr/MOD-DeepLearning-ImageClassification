import numpy as np
import os
import random
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from read_cifar import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# LBP Feature Extraction
def extract_lbp_features(images, radius=1, n_points=8, method='uniform'):
    features = []
    for image in images:
        lbp = local_binary_pattern(image, n_points, radius, method)
        # Extract histogram of LBP as the feature
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-6)  # Normalize the histogram
        features.append(hist)
    return np.array(features)

# HOG Feature Extraction
def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False):
    features = []
    for image in images:
        # Compute HOG features
        feature, _ = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize, multichannel=True)
        features.append(feature)
    return np.array(features)

# Apply image descriptor extraction (HOG and LBP)
def extract_image_descriptors(data, use_lbp=True, use_hog=True):
    descriptors = []
    for image in data:
        # Convert the 3D image (32x32x3) into 2D grayscale
        grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image_features = []

        # Use LBP if specified
        if use_lbp:
            lbp_features = extract_lbp_features([grayscale_image])
            image_features.extend(lbp_features[0])

        # Use HOG if specified
        if use_hog:
            hog_features = extract_hog_features([grayscale_image])
            image_features.extend(hog_features[0])

        descriptors.append(image_features)

    return np.array(descriptors)

# N-Fold Cross-Validation
def n_fold_cross_validation(data, labels, model, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train and test the model
        model.fit(data_train, labels_train)
        accuracy = model.evaluate(data_test, labels_test)
        accuracies.append(accuracy)

    # Return average accuracy over all folds
    return np.mean(accuracies)

# MLP Model Class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, data):
        self.a0 = data
        self.z1 = np.matmul(self.a0, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, data, labels, learning_rate):
        loss = np.mean(np.square(self.a2 - labels))

        # Gradient of loss w.r.t output
        d_loss_a2 = 2 / len(labels) * (self.a2 - labels)
        d_loss_z2 = np.matmul(d_loss_a2, self.sigmoid_derivative(self.a2))

        d_loss_w2 = np.matmul(self.a1.T, d_loss_z2) / data.shape[0]
        d_loss_b2 = np.mean(d_loss_z2, axis=0)

        # Backpropagate to the hidden layer
        d_loss_a1 = np.matmul(d_loss_z2, self.w2.T)
        d_loss_z1 = d_loss_a1 * self.a1 * (1 - self.a1)

        d_loss_w1 = np.matmul(self.a0.T, d_loss_z1) / data.shape[0]
        d_loss_b1 = np.mean(d_loss_z1, axis=0)

        # Update weights and biases
        self.w1 -= learning_rate * d_loss_w1
        self.b1 -= learning_rate * d_loss_b1
        self.w2 -= learning_rate * d_loss_w2
        self.b2 -= learning_rate * d_loss_b2

        return loss

    def fit(self, data_train, labels_train, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            predictions = self.forward(data_train)
            loss = self.backward(data_train, labels_train, learning_rate)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels_train, axis=1))
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    def evaluate(self, data_test, labels_test):
        predictions = self.forward(data_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels_test, axis=1))
        return accuracy

# Main Execution
if __name__ == "__main__":
    path = 'data/cifar-10-batches-py/'
    data, labels = read_cifar(path)
    label_names = get_label(path)

    # Normalize and prepare the data
    data = normalized(data)

    # Split the dataset into training and testing sets
    split_factor = 0.9
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split_factor)

    # Extract image descriptors (using LBP and HOG)
    use_lbp = True
    use_hog = True
    data_train_desc = extract_image_descriptors(data_train, use_lbp, use_hog)
    data_test_desc = extract_image_descriptors(data_test, use_lbp, use_hog)

    # Initialize the MLP model
    d_h = 64  # Hidden layer size
    input_dim = data_train_desc.shape[1]
    output_dim = len(np.unique(labels_train))  # Number of classes
    model = MLP(input_dim, d_h, output_dim)

    # Run N-Fold Cross-validation
    n_folds = 5
    avg_accuracy = n_fold_cross_validation(data_train_desc, labels_train, model, n_folds)
    print(f'Average Accuracy over {n_folds}-Fold Cross-Validation: {avg_accuracy * 100:.2f}%')

    # Train the model on the full training set
    model.fit(data_train_desc, labels_train, learning_rate=0.01, num_epochs=100)

    # Evaluate the model on the test set
    test_accuracy = model.evaluate(data_test_desc, labels_test)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Plot results (if needed)
    plt.plot(range(1, 101), model.train_accuracies, marker='o')
    plt.title('MLP Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
