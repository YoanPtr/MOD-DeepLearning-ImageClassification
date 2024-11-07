import sys
import os
import numpy as np
import pytest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mlp import sigmoid, sigmoid_derivative, learn_once_mse, learn_once_cross_entropy, \
                 one_hot, cross_entropy, train_mlp, run_test_mlp, run_mlp_training

# Sample setup
@pytest.fixture
def setup_data():
    np.random.seed(42)
    data_train = np.random.rand(10, 3072)  # 10 samples, CIFAR10 dimensions
    labels_train = np.random.randint(0, 10, 10)  # 10 labels from 0 to 9
    data_test = np.random.rand(5, 3072)  # 5 samples for testing
    labels_test = np.random.randint(0, 10, 5)  # 5 test labels
    d_h = 64
    learning_rate = 0.01
    num_epoch = 5
    return data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch

def test_sigmoid():
    x = np.array([0, 1, -1])
    expected = np.array([0.5, 0.73105858, 0.26894142])
    np.testing.assert_almost_equal(sigmoid(x), expected, decimal=5)

def test_sigmoid_derivative():
    x = np.array([0, 1, -1])
    expected = sigmoid(x) * (1 - sigmoid(x))
    np.testing.assert_almost_equal(sigmoid_derivative(x), expected, decimal=5)

def test_one_hot():
    labels = np.array([0, 1, 2, 1])
    result = one_hot(labels,(4,3))
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    np.testing.assert_array_equal(result, expected)

def test_cross_entropy():
    y = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    loss = cross_entropy(y, y_pred)
    assert loss > 0, "Cross entropy should be positive"

def test_learn_once_mse(setup_data):
    data_train, labels_train, *_ = setup_data
    targets = one_hot(labels_train,(len(labels_train), 10))
    w1 = np.random.rand(3072, 64)
    b1 = np.zeros((1, 64))
    w2 = np.random.rand(64, 10)
    b2 = np.zeros((1, 10))
    learning_rate = 0.01

    w1, b1, w2, b2, loss = learn_once_mse(w1, b1, w2, b2, data_train, targets, learning_rate)
    assert loss >= 0, "MSE loss should be non-negative"

def test_learn_once_cross_entropy(setup_data):
    data_train, labels_train, *_ = setup_data
    w1 = np.random.rand(3072, 64)
    b1 = np.zeros((1, 64))
    w2 = np.random.rand(64, 10)
    b2 = np.zeros((1, 10))
    learning_rate = 0.01

    w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
    assert loss >= 0, "Cross entropy loss should be non-negative"

def test_train_mlp(setup_data):
    data_train, labels_train, *_ = setup_data
    w1 = np.random.rand(3072, 64)
    b1 = np.zeros((1, 64))
    w2 = np.random.rand(64, 10)
    b2 = np.zeros((1, 10))
    learning_rate = 0.01
    num_epoch = 5

    _, _, _, _, accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    assert len(accuracies) == num_epoch, "Should have accuracy for each epoch"

def test_run_test_mlp(setup_data):
    data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch = setup_data
    w1, w2 = np.random.rand(3072, d_h), np.random.rand(d_h, 10)
    b1, b2 = np.zeros((1, d_h)), np.zeros((1, 10))

    accuracy, predictions = run_test_mlp(w1, b1, w2, b2, data_test, labels_test)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

def test_run_mlp_training(setup_data):
    data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch = setup_data

    train_accuracies, test_accuracy, _ = run_mlp_training(
        data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch
    )
    assert len(train_accuracies) == num_epoch, "Should have accuracy for each training epoch"
    assert 0 <= test_accuracy <= 1, "Test accuracy should be between 0 and 1"
