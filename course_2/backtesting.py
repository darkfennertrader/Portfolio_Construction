import sys
from typing import Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv


def constant_corr(r: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    if (delta > 1.0) or (delta < 0.0):
        raise ValueError("delta must be between 0 and 1")

    const_corr = constant_corr(r)
    sample = sample_cov(r)
    return delta * const_corr + (1 - delta) * sample


cov_estimators = {
    "sample_cov": sample_cov,
    "const_corr": constant_corr,
    "shrink_cov": shrinkage_cov,
}


class WeightingScheme(Protocol):
    def compute_weights(self, r):
        """Compute the weights for the given scheme"""


@dataclass
class GlobalMiminumVariance:
    """
    Return the weights of the GlobalminimumVariance Portfolio
    """

    cov_estimator: str
    delta: float = 0.5  # param used for shrinkage_cov function

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

    def compute_weights(self, r) -> pd.Series:
        """
        Produces the weights of the GMV portfolio given a covariance matrix of the returns
        """
        kwargs = {"delta": 0.0} if self.delta == 0.0 else {"delta": self.delta}
        est_cov = cov_estimators[self.cov_estimator](r, **kwargs)
        return self.gmv(est_cov)


class EquallyWeighted:
    """
    Returns the weights of the equally weighted portfolio
    """

    def compute_weights(self, r) -> pd.Series:
        n = len(r.columns)
        # print(r.shape, n)
        return pd.Series(1 / n, index=r.columns)


@dataclass
class EqualRiskContribution:
    cov_estimator: str
    delta: float = 0.5  # param used for shrinkage_cov function

    def target_risk_contributions(
        self, target_risk: np.ndarray, cov: pd.DataFrame
    ) -> pd.Series:
        """
        Returns the weights of the portfolio that gives you the weights such
        that the contributions to portfolio risk are as close as possible to
        the target_risk, given the covariance matrix
        """
        n = cov.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        def portfolio_vol(weights: np.ndarray, covmat: pd.DataFrame):
            """
            Computes volatility of portfolio
            """
            return (weights.T @ covmat @ weights) ** 0.5

        def risk_contribution(w: pd.Series, cov: pd.DataFrame) -> np.ndarray:
            """
            Compute the contributions to risk of the constituents of a portfolio,
            given a set of portfolio weights and a covariance matrix
            """
            total_portfolio_var = portfolio_vol(w, cov) ** 2
            # Marginal contribution of each constituent
            marginal_contrib = cov @ w
            risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
            return risk_contrib

        def msd_risk(weights, target_risk, cov):
            """
            Returns the Mean Squared Difference in risk contributions
            between weights and target_risk
            """
            w_contribs = risk_contribution(weights, cov)
            return ((w_contribs - target_risk) ** 2).sum()

        weights = minimize(
            msd_risk,
            init_guess,
            args=(target_risk, cov),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=bounds,
        )
        return weights.x

    def equal_risk_contributions(self, cov) -> pd.Series:
        """
        Returns the weights of the portfolio that equalizes the contributions
        of the constituents based on the given covariance matrix
        """
        n = cov.shape[0]
        return self.target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov)

    def compute_weights(self, r: pd.Series) -> pd.Series:
        """
        Produces the weights of the ERC portfolio given a covariance matrix of the returns
        """
        kwargs = {"delta": 0.0} if self.delta == 0.0 else {"delta": self.delta}
        est_cov = cov_estimators[self.cov_estimator](r, **kwargs)
        return self.equal_risk_contributions(est_cov)


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
class BlackLitterman:
    """
    Return the weights by implementing the classic Black-Litterman model
    """

    def as_colvec(self, x: np.array) -> np.array:  # type: ignore
        """
        this function takes either a numpy array or a numpy one-column matrix
        (i.e. a column vector) and returns the data as a column vector.
        """
        if x.ndim == 2:
            return x
        else:
            return np.expand_dims(x, axis=1)

    def implied_returns(
        self, delta: float, sigma: pd.DataFrame, w: pd.Series
    ) -> pd.Series:
        """
        This function performs reverse engineering by estimating the implied returns vector pi
        from a set of portfolio weights.
        Inputs:
        delta: Risk Aversion Coefficient (scalar)
        sigma: Variance-Covariance Matrix (N x N) as DataFrame
            w: Portfolio weights (N x 1) as Series
        Returns an N x 1 vector of Returns as Series
        """
        ir = delta * sigma.dot(w).squeeze()  # type: ignore # to get a series from a 1-column dataframe
        ir.name = "Implied Returns"
        return ir

    def proportional_prior(
        self, sigma: pd.DataFrame, tau: float, p: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns the He-Litterman (from a paper) simplified Omega.
        Assumes that Omega is proportional to the variance of the prior
        Inputs:
        sigma: N x N Covariance Matrix as DataFrame
        tau: a scalar
        p: a K x N DataFrame linking Q and Assets
        returns a P x P DataFrame, a Matrix representing Prior Uncertainties
        """
        helit_omega = p.dot(tau * sigma).dot(p.T)
        # Make a diag matrix from the diag elements of Omega
        return pd.DataFrame(
            np.diag(np.diag(helit_omega.values)), index=p.index, columns=p.index
        )

    def bl(
        self,
        w_prior: pd.Series,
        sigma_prior: pd.DataFrame,
        p: pd.DataFrame,
        q: pd.Series,
        omega: pd.DataFrame = None,  # type: ignore
        delta: float = 2.5,
        tau: float = 0.02,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        # Computes the posterior expected returns based on
        # the original black litterman reference model
        #
        # W.prior must be an N x 1 vector of weights, a Series
        # Sigma.prior is an N x N covariance matrix, a DataFrame
        # P must be a K x N matrix linking Q and the Assets, a DataFrame
        # Q must be an K x 1 vector of views, a Series
        # Omega must be a K x K matrix a DataFrame, or None
        # if Omega is None, we assume it is
        #    proportional to variance of the prior
        # delta and tau are scalars
        """
        if omega is None:
            omega = self.proportional_prior(sigma_prior, tau, p)

        # Force w.prior and Q to be column vectors
        # How many assets do we have?
        N = w_prior.shape[0]
        # And how many views?
        K = q.shape[0]
        # First, reverse-engineer the weights to get pi
        pi = self.implied_returns(delta, sigma_prior, w_prior)
        # Adjust (scale) Sigma by the uncertainty scaling factor
        sigma_prior_scaled = tau * sigma_prior
        # posterior estimate of the mean, use the "Master Formula"
        # we use the versions that do not require
        # Omega to be inverted (see previous section)
        # this is easier to read if we use '@' for matrixmult instead of .dot()
        #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
        mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(
            inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values)
        )

        # posterior estimate of uncertainty of mu.bl
        # sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
        sigma_bl = (
            sigma_prior
            + sigma_prior_scaled
            - sigma_prior_scaled.dot(p.T)
            .dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega))
            .dot(p)
            .dot(sigma_prior_scaled)
        )

        return (mu_bl, sigma_bl)

    def compute_weights(self, r):
        pass


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
