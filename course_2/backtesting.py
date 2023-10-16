import sys
from typing import Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize





def constant_corr(r: pd.DataFrame,**kwargs) -> pd.DataFrame:
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.0)
    sd = r.std()
    ccov = ccor * np.outer(sd, sd)
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)

def sample_cov(r: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def shrinkage_cov(r: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # extra params for the specific method
    delta = kwargs["delta"]
    
    const_corr = constant_corr(r)
    sample = sample_cov(r)
    return delta*const_corr + (1-delta)*sample
                  

cov_estimators = {"sample_cov": sample_cov, "const_corr": constant_corr, "shrink_cov": shrinkage_cov}


class WeightingScheme(Protocol):
    def compute_weights(self, r):
        """Compute the weights for the given scheme"""


@dataclass
class GlobalMiminumVariance:
    """
    Return the weights of the GlobalminimumVariance Portfolio
    """

    cov_estimator: str
    delta : float =  0.5 # param used for shrinkage_cov function

    def portfolio_return(self, weights, returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns

    def portfolio_vol(self, weights, covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        vol = (weights.T @ covmat @ weights) ** 0.5
        return vol

    def msr(self, riskfree_rate, er, cov):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        def neg_sharpe(weights, riskfree_rate, er, cov):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            r = self.portfolio_return(weights, er)
            vol = self.portfolio_vol(weights, cov)
            return -(r - riskfree_rate) / vol

        weights = minimize(
            neg_sharpe,
            init_guess,
            args=(riskfree_rate, er, cov),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=bounds,
        )
        return weights.x

    def gmv(self, cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return self.msr(0, np.repeat(1, n), cov)

    def compute_weights(self, r):
        """
        Produces the weights of the GMV portfolio given a covariance matrix of the returns
        """
        if (self.delta > 1.) or (self.delta < 0.):
                raise ValueError("delta must be between 0 and 1")
                
        kwargs = {"delta": 0.}  if self.delta == 0. else {"delta": self.delta}
        est_cov = cov_estimators[self.cov_estimator](r, **kwargs)
        return self.gmv(est_cov)


class EquallyWeighted:
    """
    Returns the weights of the equally weighted portfolio
    """

    def compute_weights(self, r):
        n = len(r.columns)
        #print(r.shape, n)
        return pd.Series(1 / n, index=r.columns)


@dataclass
class CapWeighted:
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """

    cap_weights: pd.Series

    def compute_weights(self, r):
        w = self.cap_weights.loc[r.index[1]]
        return w / w.sum()


@dataclass
class Backtester:
    weighting: WeightingScheme
    estimation_window: int = 60

    def run(self, r):
        """
        Backtests a given weighting scheme, given some parameters:
        r : asset returns to use to build the portfolio
        estimation_window: the window to use to estimate parameters
        weighting: the weighting scheme to use, must be a function that takes "r",
        and a variable number of keyword-value arguments
        """
        n_periods = r.shape[0]
        # return windows
        windows = [
            (start, start + self.estimation_window)
            for start in range(n_periods - self.estimation_window)
        ]
        weights = [
            self.weighting.compute_weights(r.iloc[win[0] : win[1]]) for win in windows
        ]
        # convert list of weights to DataFrame
        weights = pd.DataFrame(
            weights, index=r.iloc[self.estimation_window :].index, columns=r.columns
        )
        # return weights
        returns = (weights * r).sum(
            axis="columns", min_count=1
        )  # mincount is to generate NAs if all inputs are NAs
        return returns
