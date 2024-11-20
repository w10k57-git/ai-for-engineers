# pylint: disable=invalid-name
"""
Perceptron Module

This module implements a simple perceptron model for binary classification tasks.
It includes functions for data generation, model training, prediction, and evaluation.

The perceptron is a fundamental building block in neural networks and machine learning.
This implementation focuses on the basic perceptron algorithm for educational purposes.

Key Components:
- Data generation: Create synthetic datasets for training and testing
- Perceptron model: Implementation of the perceptron algorithm
- Utility functions: Helper functions for computations and evaluations

Functions:
- generate_data: Generate synthetic dataset with specified examples and features
- get_weighted_sum: Compute the weighted sum of features
- sigmoid: Apply the sigmoid activation function
- confusion_matrix: Calculate the confusion matrix for binary classification

Usage:
This module can be used to understand the basics of perceptron learning,
experiment with binary classification tasks, and serve as a foundation
for more complex neural network implementations.

Dependencies:
- numpy: For numerical computations
- pandas: For data manipulation and storage

Note:
This implementation is designed for educational purposes and may not be
optimized for large-scale or production use cases.
"""

import pandas as pd
import numpy as np

def generate_data(m_examples, n_features, seed=None):
    """
    Generate a synthetic dataset with specified numbers of examples and features.

    This function creates a dataset consisting of random features and binary targets. 
    The features are generated from a uniform distribution, and the targets are binary, 
    randomly selected from {0, 1}. The random number generator can be seeded for 
    reproducibility.

    Parameters
    ----------
    m_examples : int
        The number of examples to generate in the dataset.
    n_features : int
        The number of features to generate for each example.
    seed : int, optional
        A seed for the random number generator to ensure reproducibility. 
        Default is None, which means no seed will be used.

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame containing the generated features and targets. Each column represents 
        a feature named 'x0', 'x1', ..., 'x(n_features-1)', and an additional 'targets' 
        column for binary targets.

    Examples
    --------
    >>> generate_data(5, 3, seed=42)
       x0        x1        x2  targets
    0  0.374540  0.950714  0.731994        0
    1  0.598658  0.156019  0.155995        1
    2  0.058084  0.866176  0.601115        1
    3  0.708073  0.020584  0.969910        1
    4  0.832443  0.212339  0.181825        0

    Notes
    -----
    The function uses numpy's default random number generator and pandas to create the DataFrame.
    """
    generator = np.random.default_rng(seed)
    features = generator.random((m_examples, n_features))
    targets = generator.choice([0, 1], m_examples)
    data = pd.DataFrame(features, columns=[('x' + str(i)) for i in range(n_features)])
    data["targets"] = targets
    return data

def network_init(n_features, random_bias=False, seed=None):
    """
    Initialize the weights and bias for a single-layer neural network.

    This function generates a set of weights and a bias for a neural network layer with 
    a specified number of features. The weights are initialized randomly. The bias is 
    either set to zero or initialized randomly based on the `random_bias` parameter.
    A seed can be specified for the random number generator to ensure reproducibility.

    Parameters
    ----------
    n_features : int
        The number of features (inputs) for the neural network layer.
    random_bias : bool, optional
        If True, the bias is initialized randomly. Otherwise, it is set to 0.
        Default is False.
    seed : int or None, optional
        A seed for the random number generator to ensure reproducibility.
        If None, the generator is initialized without a fixed seed. 
        Default is None.

    Returns
    -------
    weights : ndarray
        A 1D array of shape (n_features,) containing the initialized weights.
    bias : float or ndarray
        The initialized bias. It is a float (0) if `random_bias` is False, 
        or a 1D array of shape (1,) with a random value if `random_bias` is True.

    Examples
    --------
    >>> network_init(3, random_bias=True, seed=42)
    (array([0.37454012, 0.95071431, 0.73199394]), array([0.59865848]))

    >>> network_init(2, random_bias=False, seed=42)
    (array([0.37454012, 0.95071431]), 0)

    Notes
    -----
    The function uses numpy's default random number generator for creating the weights
    and bias. The shape of the weights is a 1D array for ease of use in single-layer
    neural networks.
    """
    generator = np.random.default_rng(seed)
    weights = generator.random((1, n_features))[0]
    if random_bias:
        bias = generator.random((1, 1))[0]
    else:
        bias = 0
    return weights, bias

def get_weighted_sum(features, weights, bias):
    """
    Compute the weighted sum of features with the given weights and bias.

    This function calculates the dot product of the features and weights, and then adds 
    the bias to this product. It's a fundamental operation in many linear models and 
    neural networks, representing a linear combination of inputs.

    Parameters
    ----------
    features : ndarray
        An array of features (inputs). This can be a 1D array for a single set of features,
        or a 2D array for multiple sets, where each row represents a set of features.
    weights : ndarray
        An array of weights. The length of this array should match the number of features.
    bias : float or ndarray
        The bias term. Can be a scalar (if the same bias is to be added to all feature sets)
        or an array (if different biases are to be added to different feature sets).

    Returns
    -------
    ndarray
        The weighted sum of the features with the weights and bias. The shape of the return
        value depends on the input shapes. For a 1D `features` array and scalar `bias`, 
        the result is a scalar. For a 2D `features` array and scalar/vector `bias`, 
        the result is a 1D array.

    Examples
    --------
    >>> get_weighted_sum(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), 1)
    2.4

    >>> get_weighted_sum(np.array([[1, 2, 3], [4, 5, 6]]), np.array([0.1, 0.2, 0.3]), 1)
    array([2.4, 5.5])

    Notes
    -----
    The function is designed to work with NumPy arrays. Ensure that the dimensions of the
    input arrays (`features` and `weights`) match appropriately for matrix multiplication.
    """
    return np.dot(features, weights) + bias


def sigmoid(weighted_sum):
    """
    Compute the sigmoid function of the given weighted sum.

    This function applies the sigmoid activation function to a given input (weighted sum). 
    The sigmoid function is defined as 1 / (1 + exp(-x)), where x is the input. It is 
    commonly used in logistic regression and neural networks as an activation function 
    that maps any real-valued number into the range (0, 1).

    Parameters
    ----------
    weighted_sum : ndarray or scalar
        The weighted sum input to the sigmoid function. Can be a scalar, a 1D array, 
        or a 2D array.

    Returns
    -------
    ndarray or scalar
        The sigmoid of the input. The output shape is identical to the input shape.

    Examples
    --------
    >>> sigmoid(0)
    0.5

    >>> sigmoid(np.array([0, 2, -2]))
    array([0.5       , 0.88079708, 0.11920292])

    Notes
    -----
    The sigmoid function can lead to vanishing gradients when used in deep neural networks,
    as derivatives of very high or very low inputs are close to zero. It's also not zero-centered.
    """
    return 1 / (1 + np.exp(-weighted_sum))

def confusion_matrix(Y, Y_hat):
    """
    Calculate the confusion matrix for binary classification.

    The function computes the confusion matrix for a binary classifier based on the actual and predicted values. 
    The confusion matrix is a 2x2 NumPy array where each cell corresponds to the count of true positives, 
    false positives, false negatives, and true negatives.

    Parameters
    ----------
    Y : array_like
        An iterable (like a list or array) of actual binary values (0s and 1s). 
        Represents the ground truth labels.

    Y_hat : array_like
        An iterable (like a list or array) of predicted binary values (0s and 1s). 
        Represents the predicted labels from the model.

    Returns
    -------
    matrix : ndarray
        A 2x2 confusion matrix represented as a NumPy array. The matrix structure is as follows:
            [[true positives, false negatives],
             [false positives, true negatives]]

    Notes
    -----
    This function assumes that the inputs `Y` and `Y_hat` are of the same length and contain only binary values.
    It does not perform any checks for input validation.

    Examples
    --------
    >>> Y = [1, 0, 1, 0, 1]
    >>> Y_hat = [1, 1, 1, 0, 0]
    >>> confusion_matrix(Y, Y_hat)
    array([[2., 2.],
           [1., 1.]])
    """

    matrix = np.zeros((2, 2))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, j in zip(Y, Y_hat):
        if i == 1 and i == j:
            tp  += 1
        elif i == 1:
            fn += 1
        elif i == 0 and i == j:
            tn += 1
        else:
            fp += 1
    matrix[0, 0] = tp
    matrix[1, 0] = fp
    matrix[0, 1] = fn
    matrix[1, 1] = tn
    return matrix

def forward_prop(X, weights, bias, threshold=0.5):
    """
    Perform forward propagation for the perceptron model.

    This function computes the weighted sum, applies the sigmoid activation function,
    and determines the binary prediction for each example in the input data.

    Parameters
    ----------
    X : array-like
        Input features, shape (m_examples, n_features).
    weights : array-like
        Model weights, shape (n_features,).
    bias : float
        Bias term of the model.
    threshold : float, optional
        Decision threshold for binary classification. Default is 0.5.

    Returns
    -------
    Y_hat : list
        Binary predictions for each example.
    weighted_sums : list
        Weighted sums for each example before activation.
    sigmoids : list
        Sigmoid activation outputs for each example.

    Notes
    -----
    This function assumes that the necessary utility functions (get_weighted_sum, sigmoid)
    are defined elsewhere in the module.
    """
    weighted_sums = []
    sigmoids = []
    Y_hat = []
    m_examples = X.shape[0]
    for i in range(m_examples):
        example = X[i,:]
        weighted_sum = get_weighted_sum(example, weights, bias)
        weighted_sums.append(weighted_sum)
        sigm = sigmoid(weighted_sum)
        sigmoids.append(sigm)
        if sigm > threshold:
            y_hat = 1
        else:
            y_hat = 0
        Y_hat.append(y_hat)
    return Y_hat, weighted_sums, sigmoids

def cross_entropy(Y, sigmoids):
    """
    Calculate the cross-entropy loss between true labels and predicted probabilities.

    This function computes the binary cross-entropy loss, which is a common loss function
    for binary classification problems. It measures the performance of a classification
    model whose output is a probability value between 0 and 1.

    Parameters
    ----------
    Y : array-like
        True binary labels (0 or 1), shape (m_examples,).
    sigmoids : array-like
        Predicted probabilities from the sigmoid activation, shape (m_examples,).

    Returns
    -------
    list
        Cross-entropy loss for each example.

    Notes
    -----
    The cross-entropy loss is calculated as:
    -[y * log(p) + (1 - y) * log(1 - p)]
    where y is the true label and p is the predicted probability.

    This implementation uses log base 10 for the calculations.
    """
    cost = []
    m_examples = len(Y)
    for i in range(m_examples):
        y = Y[i]
        sigmoid_value = sigmoids[i]
        cross_entropy_loss = -(y*np.log10(sigmoid_value) + (1-y)*np.log10(1-sigmoid_value))
        cost.append(cross_entropy_loss)
    return cost
