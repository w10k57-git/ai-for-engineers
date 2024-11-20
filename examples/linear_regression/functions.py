"""
This module contains utility functions for generating synthetic data,
performing linear regression, and calculating associated metrics.

Functions:
    generate_data: Generate synthetic data for linear regression.
    linear_function: Calculate linear function y = ax + b.
    cost_function: Calculate mean squared error cost for linear model.
    gradient_calc: Calculate gradients for linear model cost function.

The module provides tools for creating test datasets, evaluating linear models,
and implementing gradient descent optimization for linear regression tasks.
"""

def linear_function(x, a, b):
    """
    Calculate linear function y = ax + b.
    """
    y = a * x + b
    return y

def cost_function(x, y, a, b):
    """
    Calculate mean squared error cost for linear model.
    """
    m = len(x)
    partial_sum = (linear_function(x, a, b) - y) ** 2
    total_cost = sum(partial_sum) / m
    return total_cost

def gradient_calc(x, y, a, b):
    """
    Calculate gradients for linear model cost function.
    """
    m = len(x)
    sum_agrad = 0
    sum_bgrad = 0
    for i in range(m):
        partial_sum_agrad = (linear_function(x[i], a, b) - y[i]) * x[i]
        partial_sum_bgrad = linear_function(x[i], a, b) - y[i]
        sum_agrad += partial_sum_agrad
        sum_bgrad += partial_sum_bgrad
    gradient_a = 2 * sum_agrad / m
    gradient_b = 2 * sum_bgrad / m
    return gradient_a, gradient_b

def gradient_descent(x, y, a0=0, b0=0,
                     alpha=0.1, n_iter=100, print_res=0):
    """
    Perform gradient descent for linear model optimization.
    """
    a = a0
    b = b0
    cost_history = []
    coeff_history = []
    for i in range(n_iter):
        gradient_a, gradient_b = gradient_calc(x, y, a, b)
        a = a - alpha * gradient_a
        b = b - alpha * gradient_b
        mse = cost_function(x, y, a, b)
        cost_history.append(mse)
        coeff_history.append((a, b))
        if print_res == 1:
            print(f'Cost at iter no. {i}: {mse:.2f}')
    return a, b, cost_history, coeff_history

def prediction(x, a, b):
    """
    Predict values using linear model y = ax + b.
    """
    y_pred = a * x + b
    return y_pred
