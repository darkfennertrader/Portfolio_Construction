import sys
from typing import Protocol, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

weighting = Callable[[pd.Series], pd.Series]


class EquallyWeighted:
    """
    Returns the weights of the equally weighted portfolio
    """

    def __call__(self, r):
        n = len(r.columns)
        return pd.Series(1 / n, index=r.columns)


@dataclass
class CapWeighted:
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """

    cap_weights: pd.Series

    def __call__(self, r):
        w = self.cap_weights.loc[r.index[1]]
        return w / w.sum()


@dataclass
class Backtester:
    weighting_scheme: weighting
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
        # print(r.shape[0])
        # return windows
        windows = [
            (start, start + self.estimation_window)
            for start in range(n_periods - self.estimation_window)
        ]
        # print(windows)

        weights = [self.weighting_scheme(r.iloc[win[0] : win[1]]) for win in windows]
        # convert list of weights to DataFrame
        weights = pd.DataFrame(
            weights, index=r.iloc[self.estimation_window :].index, columns=r.columns
        )
        # return weights
        returns = (weights * r).sum(
            axis="columns", min_count=1
        )  # mincount is to generate NAs if all inputs are NAs
        return returns
