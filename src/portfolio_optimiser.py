"""
Given several stocks, find the portfolio weights that produce the highest Sharpe Ratio,
while enforcing constraints such as no short selling and a maximum 30% allocation to any one stock.

0 ≤ weight ≤ 0.30 per stock allocation
"""

import numpy as np
import pandas as pd

from scipy.optimize import minimize                 # optimisation algorithm. requires historical returns 


def optimise_portfolio(prices: pd.DataFrame) -> dict:
    """
    Find portfolio weights that maximise Sharpe ratio while subjective to 30% allocation per stock.

    Optimising based on returns.
    """
    returns = prices.pct_change().dropna()

    mean_returns = returns.mean() * 252             # calculate expected annual return
    covariance_matrix = returns.cov() * 252         # covariance measures how returns of two assets move relative to eachother (how stocks returns move relative to one another). positive covariance = stock A goes up, stock B tends to go up. negative covariance = stock A goes up, stock B tends to go down. closer to 0 means stock moves independently of eachother.

    num_assets = len(prices.columns)

    def portfolio_return(weights: np.ndarray) -> float:
        """
        Calculate expected portfolio return.

        Takes two arrays and returns one numpy floating point value.
        """
        return np.sum(mean_returns * weights)       # produces a numpy scalar by combining elements into one value. numpy multiplies corresponding elements       

    def portfolio_volatility(weights: np.ndarray) -> np.float64 :
        """
        Calculate portfolios annualised volatility using portfolios weights and covariance matrix.
        """
        return np.sqrt(weights.T @ covariance_matrix @ weights)    # @ = matrix multiplication
    
    def negative_sharpe_ratio(weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio (volatility) for a given portfolio.
        """
        portfolio_ret = portfolio_return(weights)
        portfolio_vol = portfolio_volatility(weights)

        return -(portfolio_ret / portfolio_vol)                     # returns a negative value to minimise negative sharpe, therefore maximising sharpe value
    
    constraints = [{                                                # keys that scipy expects
        "type": "eq",                                               # eq = equality constraint (condition must equal 0)
        "fun": lambda weights: np.sum(weights) - 1                  # fun = function that defines constraint. constraint must = 0 as all weights would add to 1 (1-1=0)
    }]

    bounds = [(0, 0.30)] * num_assets                               # creates a list of tuples containing len(num_assets) rows each containing (0, 0.30) repping min and max allocation respectively

    initial_weights = np.array([1 / num_assets] * num_assets)       # starting point for optimiser. starts with an equally weighted portfolio

    result = minimize(negative_sharpe_ratio, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)       # minimize(), from scipy, searches for input values that make a function as small as possible. trying to minimise negative_sharpe_ratio here. SLSQP (sequential least squares programming) = optimisation algo to use. this algo handles both bounds and constraints

    optimal_weights = result.x          # result object contains properties such as x (x contains best input values found by optimiser. e.g array([0.30, 0.18, 0.22, 0.30]) aka optimised portfolio weights), fun, success, message

    return {
        "weights": dict(zip(prices.columns, optimal_weights)),      # zip pairs values and creates a tuple. turned into a dict then
        "sharpe_ratio": -result.fun,                                # value of function being minimised (a float)
        "expected_return": portfolio_return(optimal_weights),
        "volatility": portfolio_volatility(optimal_weights),
        "success": result.success                   # boolean. tells u if scipy believes if optimisation has successfully converged
    }
    
