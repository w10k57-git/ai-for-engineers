"""
Utility functions for visualizing MNIST dataset.
Functions:
    show_random_images: Display random images from dataset with labels
    plot_class_distributions: Plot class distribution in train and test sets

"""

import matplotlib.pyplot as plt
import numpy as np


def show_random_images(X, Y, num_images=25):
    """
    Display a grid of random images from the dataset with their corresponding labels.

    Parameters:
    X (array-like): Input images array
    Y (array-like): Labels array
    num_images (int, optional): Number of random images to display. Default is 25.

    Returns:
    None: Displays a matplotlib figure with random images from the dataset
    """
    # Create a shuffled index array
    shuffled_indices = np.arange(len(X))
    np.random.shuffle(shuffled_indices)

    # Select a subset of shuffled indices
    selected_indices = shuffled_indices[:num_images]

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(selected_indices):
        plt.subplot(5, 5, i + 1)
        # Reshape if necessary, e.g., for MNIST
        plt.imshow(X[idx].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f'Class: "{Y[idx]}"')
        plt.xticks([])  # Hide x-axis tick marks
        plt.yticks([])  # Hide y-axis tick marks

    plt.tight_layout()
    plt.show()

def plot_class_distributions(ytrain, ytest, bar_width=0.35, figsize=None):
    """
    Plot the distributions of classes in the training and test sets.

    Parameters:
    ytrain (array-like): Labels for the training set.
    ytest (array-like): Labels for the test set.
    bar_width (float, optional): Width of the bars. Default is 0.35.
    figsize (tuple, optional): Size of the figure. Default is (12, 6).
    """
    unique_train, counts_train = np.unique(ytrain, return_counts=True)
    unique_test, counts_test = np.unique(ytest, return_counts=True)

    plt.figure(figsize=figsize)
    index_train = np.arange(len(unique_train))
    index_test = np.arange(len(unique_test))

    plt.bar(index_train, counts_train, bar_width,
            label='Training Set', color='skyblue')
    plt.bar(index_test + bar_width, counts_test, bar_width,
            label='Test Set', color='salmon')

    plt.xlabel('Class')
    plt.ylabel('Number of occurrences')
    plt.title('Number of occurrences per class in MNIST sets')
    plt.xticks(index_train + bar_width / 2, unique_train)
    plt.legend()

    plt.show()
