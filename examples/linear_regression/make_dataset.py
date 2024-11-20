"""
This module provides functions for generating synthetic datasets
for linear and polynomial regression analysis.
"""

import numpy as np


def generate_linear_data(
    m_samples,
    x_max=1,
    a_range=10,
    b_range=1,
    noise=0.2,
    coeff=False,
    seed=18
):
    """
    Generate synthetic data for linear regression.

    Returns x and y arrays for linear regression analysis.
    """
    rg = np.random.default_rng(seed=seed)
    a_data = rg.uniform(-a_range, a_range)
    b_data = rg.uniform(-b_range, b_range)
    x_features = rg.uniform(0, x_max, m_samples)
    noise_stddev = rg.uniform(noise / 2, noise)
    y_features = (
        a_data * x_features + b_data + np.random.normal(0, noise_stddev, m_samples)
    )
    if coeff:
        print(
            f'Coefficients used to generate data: \n'
            f'a = {a_data:.2f} \nb = {b_data:.2f}'
        )
    return x_features, y_features


def generate_polynomial_data(
    m_samples,
    degree,
    x_max=1,
    coeff_range=10,
    noise=0.2,
    coeff=False,
    seed=18
):
    """
    Generate synthetic data for polynomial regression.

    Returns x and y arrays for polynomial regression analysis.
    """
    rg = np.random.default_rng(seed=seed)
    coefficients = rg.uniform(-coeff_range, coeff_range, degree + 1)
    x_features = rg.uniform(0, x_max, m_samples)
    noise_stddev = rg.uniform(noise / 2, noise)

    y_features = np.polyval(coefficients, x_features) + rg.normal(0, noise_stddev, m_samples)

    if coeff:
        print('Coefficients used to generate data:')
        for i, c in enumerate(coefficients):
            print(f'a{i} = {c:.2f}')

    return x_features, y_features
