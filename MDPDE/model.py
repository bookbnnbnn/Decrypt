import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from .calculation import *
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

class WEB:
    def __init__(
            self, 
            X: np.ndarray,
            y: np.ndarray,
            ) -> None:
        self.X = X
        self.y = y


    def paramters_initial(self)-> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Initializes the parameters for the model.

        Returns
        ----------
        Tuple containing dictionaries of initialized parameters

        mus_initial: Dict[str, np.ndarray]
            Initial values of mus
        sigmas_initial: Dict[str, np.ndarray]
            Initial values of sigmas
        self.betas_initial: Dict[str, np.ndarray]
            Initial values of betas
        """

        mu_initial = []
        self.mus_initial = np.median(mu_initial, axis=0)
        self.sigmas_initial = calculate_sigmas(self.X, self.y, self.mus_initial, initial=True)
        
        self.betas_initial = caculate_betas_MDPDE(self.X, self.y, weights)

        return self.mus_initial, self.sigmas_initial, self.betas_initial

    def MDPDE_iter(
            self, 
            max_iter: int = 3, 
            alpha: float = 0.1, 
            gamma: float = 0.1, 
            tol: float = 1e-15, 
            patience: int = 3, 
            verbose: int = 1, 
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Runs the iterative algorithm to estimate parameters.

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations.
        alpha: float
            Alpha hyperparameter.
        gamma: float
            Gamma hyperparameter.
        tol: float
            Tolerance for convergence.
        patience: int
            Number of iterations without improvement to tolerate before stopping.
        verbose: int
            If 0, show nothing. 
            If 1, display iteration progress. 
            If 2, display iteration progress and beta difference.

        Returns
        ----------
            Tuple containing the estimated betas and the mean beta difference.
        """
        # Initialize variables
        self.betas_MDPDE = self.betas_initial
        self.sigmas_MDPDE  = self.sigmas_initial

        self.beta_differences_histories = []

        cur_beta = self.betas_initial
        least_difference = np.inf
        iter_num = 0

        for i in tqdm(range(max_iter), disable = True if verbose == 0 else False):
            # Iterate the algorithm
            self.weights = caculate_weights(self.X, self.y, self.betas_MDPDE, self.sigmas_MDPDE)
            self.betas_MDPDE = caculate_betas_MDPDE(self.X, self.y, self.weights)
            self.sigmas_MDPDE = calculate_sigmas(self.X, self.y, self.betas_MDPDE, self.weights, alpha)

            # Calculate the difference between the current betas and the previous betas

            beta_difference = max(np.sum((self.betas_MDPDE - cur_beta)**2, axis=0))

            if verbose == 2:
                logging.info(f"iteration {i} finished with difference: {beta_difference}")

            iter_num += 1
            self.beta_differences_histories.append(beta_difference)

            # Check convergence criteria and update variables
            if beta_difference > tol and iter_num < (patience + 1):
                if beta_difference < least_difference:
                    least_difference = beta_difference
                    iter_num = 0
                cur_beta = self.betas_MDPDE
            else:
                self.betas_MDPDE = cur_beta
                break

        return self.betas_MDPDE, self.beta_differences_histories