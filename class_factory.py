from typing import Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd


class Strategy(Protocol):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Backtester class uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def compute_weights(self, r):
        pass


class EquallyWeighted:
    def compute_weights(self, r):
        n = len(r.columns)
        return pd.Series(1 / n, index=r.columns)


class Backtester:
    def __init__(self, weighting_scheme: Strategy) -> None:
        self.weighting_scheme = weighting_scheme

    def backtest_ws(self, r, estimation_window=60, **kwargs):
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
            (start, start + estimation_window)
            for start in range(n_periods - estimation_window)
        ]
        weights = [
            self.weighting_scheme.compute_weights(r.iloc[win[0] : win[1]], **kwargs)
            for win in windows
        ]
        # convert list of weights to DataFrame
        weights = pd.DataFrame(
            weights, index=r.iloc[estimation_window:].index, columns=r.columns
        )
        # return weights
        returns = (weights * r).sum(
            axis="columns", min_count=1
        )  # mincount is to generate NAs if all inputs are NAs
        return returns
