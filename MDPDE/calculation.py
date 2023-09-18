import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple, List

def calculate_sigmas(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    betas: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    initial: bool = False
) -> np.ndarray:
    """
    Calculates the sigmas based on the given data and betas.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas: np.ndarray
        beta arrays.
    initial: bool = False
        Flag indicating whether initial sigmas are being calculated.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated sigmas.

    """

    sigma_initial = []

    if initial:
        betas = [betas] * len(y_tilde_all)

    # Calculate sigmas for each data point
    for X_tilde, y_tilde, beta, weight in zip(X_tilde_all, y_tilde_all, betas, weights):
        residual = (y_tilde - X_tilde @ beta)
        nominator = np.sum(weight * residual ** 2)
        denominator = np.sum(weight) - alpha / (1 + alpha) ** (3 / 2)
        sigma_initial.append(nominator / denominator)

    return np.array(sigma_initial)



def caculate_betas_MDPDE(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    weight_all: np.ndarray
) -> np.ndarray:
    """
    Calculates the weighted beta values using the given data and weights.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    weights: np.ndarray
        weight arrays.

    Returns
    ----------
    np.ndarray
        Array containing the calculated weighted beta values.

    """

    beta_tilde = []

    # Calculate weighted beta for each data point
    for weight, X_tilde, y_tilde in zip(weight_all, X_tilde_all, y_tilde_all):
        beta_tilde.append(np.linalg.inv(weight * X_tilde.T @ X_tilde) @ (weight * X_tilde.T @ y_tilde))

    return np.array(beta_tilde)


def caculate_weights(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    beta_em_all: np.ndarray,
    sigma_all: np.ndarray,
    alpha: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the weights and lambdas using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas_em: np.ndarray
        beta_em arrays.
    sigmas: np.ndarray
        sigma_mle arrays.
    alpha: float
        Alpha parameter value.

    Returns:
    ----------
    Tuple containing the calculated weights.

    weights: np.ndarray
        The weight of each data point.
    """

    weight = np.array([np.ones(len(y_tilde_all[0]))] * len(y_tilde_all))

    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_em_all, sigma_all)):
        X_tilde, y_tilde, beta_em, sigma = elements
        
        # Calculate weight
        if alpha is not None:
            exponential_w = np.exp(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta_em)**2)
            # weight[idx] =  exponential_w / np.sum(exponential_w) * len(y_tilde) 
            weight[idx] =  exponential_w

    return weight
