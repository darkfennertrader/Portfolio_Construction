import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:.6f}".format
plt.rcParams["figure.figsize"] = (10, 6)


def pre_processing_hfi():
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True
    )
    # some Pre-Processing
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    return hfi


def pre_processing_ind():
    ind = (
        pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)
        / 100
    )
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind30_size():
    """ """
    ind = pd.read_csv("data/ind30_m_size.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind30_nfirms():
    """
    Loads and format the Ken French 30 Industry Portfolios Value weighted monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind


class Metrics:
    """
    This class computes common metrics from a time series of asset returns
    with these data
    """

    def __init__(self, starting_amount: int = 1):
        self.starting_amount = starting_amount

    def portfolio_return(self, weights: np.array, returns: pd.Series):
        """
        Computes weighted average return of portfolio: wT * R
        """
        return weights.T @ returns

    def portfolio_vol(self, weights: np.array, covmat: pd.DataFrame):
        """
        Computes volatility of portfolio
        """
        return (weights.T @ covmat @ weights) ** 0.5

    def annualized_rets(self, series: pd.Series, periods_per_year=12):
        """
        Annualize the returns of a series by inferring the periods per year
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """
        compounded_growth = (1 + series).prod()
        n_periods = series.shape[0]
        return compounded_growth ** (periods_per_year / n_periods) - 1

    def annualized_vol(self, series: pd.Series, periods_per_year=12):
        """
        Annualize the volatility of a series by inferring the periods per year
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """
        return series.std() * (periods_per_year**0.5)

    def sharpe_ratio(self, series: pd.Series, riskfree_rate=0.03, periods_per_year=12):
        """
        Computes the annualized Sharpe Ratio of a series
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """

        rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
        excess_ret = series - rf_per_period
        ann_ex_ret = self.annualized_rets(excess_ret, periods_per_year)
        ann_vol = self.annualized_vol(series, periods_per_year)
        return ann_ex_ret / ann_vol

    def skewness(self, series: pd.Series):
        """
        Compute the skewness of the supplied Series/Dataframe
        """
        demeaned_data = series - series.mean()
        # population std dddof=0
        sigma_data = series.std(ddof=0)
        exp = (demeaned_data**3).mean()
        return exp / sigma_data**3

    def kurtosis(self, series: pd.Series):
        """
        Compute the kurtosis of the supplied Series/Dataframe
        """
        demeaned_data = series - series.mean()
        # population std dddof=0
        sigma_data = series.std(ddof=0)
        exp = (demeaned_data**4).mean()
        return exp / sigma_data**4

    def semideviation_daily_ret(self, series: pd.Series):
        """
        Compute the negative semideviation of the supplied Series when returns are close to zero
        like in the case of daily returns
        """
        is_negative = series < 0
        return series[is_negative].std(ddof=0)

    def semideviation(self, series: pd.Series):
        """
        Compute the negative semideviation of the supplied Series
        """
        excess = series - series.mean()
        excess_negative = excess[excess < 0]
        excess_negative_square = excess_negative**2
        n_negative = (excess < 0).sum()
        return (excess_negative_square.sum() / n_negative) ** 0.5

    def is_normal(self, series: pd.Series, level=0.01):
        """
        Applies the Jarque-Bera test to determine if a Series is Normal at a specified level of confidence
        """
        statistic, p_value = jarque_bera(series)
        return p_value > level

    def historical_var(self, series: pd.Series, level=5):
        """
        computes VaR based on historical data (non-parametric) over the data frequency.
        There is a 5% (level=5) you are going to lose the "calculated amount" in a certain period
        (given by the data frequency)
        """
        return -np.percentile(series, level)

    def cornish_fisher_var(self, series: pd.Series, level=5, modified=True):
        """
        computes VaR based on semi-parametric Cornish-Fisher method over the data frequency.
        PARAMS:
        - modified: True: uses the Cornish-Fisher modification
                    False: uses the standard Gaussian distribution
        There is a 5% (level=5) you are going to lose the "calculated amount" in a certain period
        (given by the data frequency)
        """
        z_score = norm.ppf(level / 100)
        if modified:
            s = self.skewness(series)
            k = self.kurtosis(series)
            z_score = (
                z_score
                + (z_score**2 - 1) * s / 6
                + (z_score**3 - 3 * z_score) * (k - 3) / 24
                - (2 * z_score**3 - 5 * z_score) * (s**2) / 36
            )

        return -(series.mean() + z_score * series.std(ddof=0))

    def historical_cvar(self, series: pd.Series):
        """
        computes CVaR based based on historical data over the data frequency.
        """
        is_beyond = series <= -self.historical_var(series)

        return -series[is_beyond].mean()

    def drawdown(self, data: pd.Series) -> pd.DataFrame:
        """
        Computes the drawdown of a time series
        """
        # fig, axs = plt.subplots(2)
        # compute wealth index
        wealth_index = self.starting_amount * (1 + data).cumprod()
        # print(wealth_index)
        # axs[0].plot(wealth_index.index.to_timestamp(), wealth_index)
        # compute previuos peaks
        previuos_peaks = wealth_index.cummax()
        # axs[0].plot(previuos_peaks.index.to_timestamp(), previuos_peaks)
        ## compute drawdown
        drawdowns = (wealth_index - previuos_peaks) / previuos_peaks
        # axs[1].plot(drawdowns.index.to_timestamp(), drawdowns)
        # print(f"max drawdown is {drawdowns.min()*100:.2f}% located at {drawdowns.idxmin()}\n")

        return pd.DataFrame(
            {"Wealth": wealth_index, "Peaks": previuos_peaks, "Drawdown": drawdowns}
        )

    def minimize_vol(self, target_return: float, rets: pd.Series, cov: pd.DataFrame):
        """
        Minimizes volatility for a given target return
        subject to the constraints of the Mean-Variance Portfolio
        """
        n_assets = rets.shape[0]

        # initial guess
        init_guess = np.repeat(1 / n_assets, n_assets)

        # no shorting (it's a tuple of tuples)
        bounds = ((0, 1.0),) * n_assets

        # for a given target return the portfolio return is equal
        return_is_target = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda weights, rets: target_return
            - self.portfolio_return(weights, rets),
        }

        # weights su to 1
        weights_sum_to_one = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        results = minimize(
            self.portfolio_vol,
            init_guess,
            args=(cov,),
            method="SLSQP",
            options={"disp": False},
            constraints=(return_is_target, weights_sum_to_one),
            bounds=bounds,
        )
        return results.x

    def _optimal_weights(
        self, n_points: int, rets: pd.Series, cov: pd.DataFrame
    ) -> list[np.array]:
        """
        list of weights to run the optimizer on to minimize volatility
        for a given target return

        RiskFree rate  + ER + COV -> weights
        """
        # generate target return between minimun and maximum target returns
        target_rs = np.linspace(rets.min(), rets.max(), n_points)
        # find the weights that minimizes volatility given a certain target return
        weights = [
            self.minimize_vol(target_return, rets, cov) for target_return in target_rs
        ]
        return weights

    def _neg_sharpe_ratio(
        self,
        weights: pd.Series,
        riskfree_rate: float,
        rets: pd.Series,
        cov: pd.DataFrame,
    ):
        """
        Returns the negative of the Sharpe Ratio
        """
        r = self.portfolio_return(weights, rets)
        vol = self.portfolio_vol(weights, cov)

        return -((r - riskfree_rate) / vol)

    def maximum_sharpe_ratio(
        self, riskfree_rate: float, rets: pd.Series, cov: pd.DataFrame
    ):
        """
        Computes the Maximum Sharpe Ratio Portfolio
        subject to the constraints of the Mean-Variance Portfolio
        """
        n_assets = rets.shape[0]
        # initial guess
        init_guess = np.repeat(1 / n_assets, n_assets)
        # no shorting (it's a tuple of tuples)
        bounds = ((0, 1.0),) * n_assets
        # weights su to 1
        weights_sum_to_one = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        results = minimize(
            self._neg_sharpe_ratio,
            init_guess,
            args=(
                riskfree_rate,
                rets,
                cov,
            ),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_one),
            bounds=bounds,
        )
        return results.x

    def gmv(self, cov: pd.DataFrame):
        """
        Returns the weights of the Global Minimum Volatility Portfolio
        given the covariance matrix. The trick used here is to feed the GMV optimizer with
        min and max of Returns equal to each other. In this case the optimizer will try to minimize the volatility
        given that all Returns are equal to each other.
        """
        n_assets = cov.shape[0]
        # I don't care what it is since it is not used anyway
        rfr = 0
        # I don't care the number used for Returns as long as it is the same for every asset
        returns = np.repeat(1, n_assets)
        return self.maximum_sharpe_ratio(riskfree_rate=rfr, rets=returns, cov=cov)

    def plot_ef(
        self,
        n_points: int,
        returns: pd.Series,
        cov: pd.DataFrame,
        show_cml=False,
        riskfree_rate: float = 0.0,
        show_ew=False,
        show_gmv=False,
    ) -> None:
        """
        Plots the multi-asset efficient frontier
        """
        weights = self._optimal_weights(n_points, returns, cov)
        # print(weights)
        rets = [self.portfolio_return(w, returns) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        eff_frontier = pd.DataFrame({"R": rets, "Vol": vols})
        ax = eff_frontier.plot.line(x="Vol", y="R", style=".-")

        if show_ew:
            n_assets = returns.shape[0]
            w_ew = np.repeat(1 / n_assets, n_assets)
            r_ew = self.portfolio_return(w_ew, returns)
            vol_ew = self.portfolio_vol(w_ew, cov)
            # display EW (Equal Weighted) Portfolio
            ax.annotate("EW", (vol_ew, r_ew))
            ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10)

        if show_gmv:
            w_gmv = self.gmv(cov)
            r_gmv = self.portfolio_return(w_gmv, returns)
            vol_gmv = self.portfolio_vol(w_gmv, cov)
            # display GMV (Global minimum Variance) Portfolio
            ax.annotate("GMV", (vol_gmv, r_gmv))
            ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)

        if show_cml:
            # Add CML plot
            ax.set_xlim(left=0)
            riskfree_rate = riskfree_rate
            w_msr = self.maximum_sharpe_ratio(riskfree_rate, returns, cov)
            r_msr = self.portfolio_return(w_msr, returns)
            vol_msr = self.portfolio_vol(w_msr, cov)
            # display CML (Capital Market Line) Portfolio
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.annotate("MSR", (vol_msr, r_msr))
            ax.plot(
                cml_x,
                cml_y,
                color="green",
                marker="o",
                linestyle="dashed",
                markersize=10,
            )

        return ax

    def run_cppi(
        self,
        risky_r: pd.Series | pd.DataFrame,
        safe_r=None,
        m=3,
        start=1000,
        floor=0.8,
        riskfree_rate=0.03,
        drawdown_constraint=None,
    ):
        """
        Run a backtest of the CPPI strategy given a set of returns for the risky assets
        Returns a dictionary containing: Asset Value History, risk Budget History, Risk Weight History.
        Steps for CPPI Algorithm:
            1. Cushion - (Asset Value minus Floor Value)
            2. Compute an allocation to Safe and Risky Assets --> m*risk_budget
            3. Recompute the Asset Values based on returns
        """
        # setup the CPPI params
        dates = risky_r.index
        n_steps = len(dates)
        account_value = start
        floor_value = start * floor
        peak = start  # need for drawdown constraint

        if isinstance(risky_r, pd.Series):
            risky_r = pd.DataFrame(risky_r, columns=["R"])

        if safe_r is None:
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r[:] = riskfree_rate / 12

        # setup some dataframes to save intermediate results
        account_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)

        # CPPI algorithm implementation
        for step in range(n_steps):
            # drawdown constraint
            if drawdown_constraint is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown_constraint)
            cushion = (account_value - floor_value) / account_value
            # weight allocation to the risky asset
            risky_w = m * cushion
            # leverage without borrowing
            risky_w = np.minimum(risky_w, 1)
            # no shorting
            risky_w = np.maximum(risky_w, 0)
            # weight allocation to the safe asset
            safe_w = 1 - risky_w
            # money allocation
            risky_alloc = account_value * risky_w
            safe_alloc = account_value * safe_w

            # update the account value at the end of the step
            account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (
                1 + safe_r.iloc[step]
            )

            # save values to look and plot their history
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value

        # reference for investing only in risky assets
        risky_wealth = start * (1 + risky_r).cumprod()

        backtest_results = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth,
            "Risky Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_r": risky_r,
            "safe_r": safe_r,
        }

        return backtest_results

    def summary_stats(
        self, data: pd.Series | pd.DataFrame, riskfree_rate=0.03
    ) -> pd.DataFrame:
        """
        Returns a Dataframe with all the relevant metrics for every asset in the data
        """

        def _stats(r: pd.Series, riskfree_rate=0.03) -> dict:
            """
            Returns a dict that contains aggregated summary stats for returns in the Series
            """
            ann_r = self.annualized_rets(r, periods_per_year=12)
            ann_vol = self.annualized_vol(r, periods_per_year=12)
            ann_sr = self.sharpe_ratio(
                r, riskfree_rate=riskfree_rate, periods_per_year=12
            )
            dd = self.drawdown(r)
            skew = self.skewness(r)
            kurt = self.kurtosis(r)
            cf_var5 = self.cornish_fisher_var(r, modified=True)
            hist_cvar5 = self.historical_cvar(r)

            return {
                "Ann. Return": ann_r,
                "Ann. Volatility": ann_vol,
                "Skewness": skew,
                "Kurtosis": kurt,
                "Corn-Fisher VaR (5%)": cf_var5,
                "Hist. CVar (5%)": hist_cvar5,
                "Sharpe Ratio": ann_sr,
                "Max Drawdown": dd["Drawdown"].min(),
                # "Max DD date": dd.idxmin()["Drawdown"]
            }

        # Returns a Series (agg applies the function to every column of the data (Dataframe)
        res = (
            data.agg(_stats).to_frame().to_dict()[0]
        )  # returned Series is converted to dict
        # create metrics for every asset
        idx_list = []
        value_list = []
        for idx, values in res.items():
            idx_list.append(idx)
            value_list.append(values)

        return pd.DataFrame(index=idx_list, data=value_list)

    def gbm(
        self,
        n_years=10,
        n_scenarios=1000,
        mu=0.07,
        sigma=0.15,
        steps_per_year=12,
        s_0=100.0,
        prices=True,
    ) -> pd.DataFrame:
        """
        Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
        :param n_years:  The number of years to generate data for
        :param n_paths: The number of scenarios/trajectories
        :param mu: Annualized Drift, e.g. Market Return
        :param sigma: Annualized Volatility
        :param steps_per_year: granularity of the simulation
        :param s_0: initial value
        :return: a numpy array of n_paths columns and n_years*steps_per_year rows
        """
        dt = 1 / steps_per_year
        n_steps = int(n_years * steps_per_year) + 1
        # without discretization error ...
        rets_plus_1 = np.random.normal(
            loc=(1 + mu) ** dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios)
        )
        rets_plus_1[0] = 1
        # from return to prices
        # prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
        ret_val = (
            s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
        )
        return ret_val

    def show_gbm(self, n_scenarios, mu, sigma) -> None:
        """
        Draw the results of an asset price evolution under the Gemetric Brownian Motion Model
        """
        s_0 = 100
        prices = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
        ax = prices.plot(
            legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(12, 5)
        )
        ax.axhline(y=s_0, ls=":", color="black")
        ax.set_ylim(top=400)
        # draw a dot at the origin
        ax.plot(0, s_0, marker="o", color="darkred", alpha=0.2)

    def show_cppi(
        self,
        n_scenarios=50,
        mu=0.07,
        sigma=0.15,
        m=3,
        floor=0.0,
        riskfree_rate=0.03,
        y_max=100,
    ) -> None:
        """
        Plot the results of a Monte Carlo simulation of CPPI
        """
        start = 100
        sim_rets = self.gbm(
            n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12
        )
        risky_rets = pd.DataFrame(sim_rets)
        # run the backtest
        btr = self.run_cppi(
            risky_r=pd.DataFrame(risky_rets),
            riskfree_rate=riskfree_rate,
            m=m,
            start=start,
            floor=floor,
        )
        wealth = btr["Wealth"]
        # scale max value: effect is to zoom in/out a portion of the plot
        y_max = (
            wealth.values.max() * y_max / 100
        )  # scale max value: effect is to zoom in/out a portion of the plot
        # calculate terminal wealth stats
        terminal_wealth = wealth.iloc[-1]
        tw_mean = terminal_wealth.mean()
        tw_median = terminal_wealth.median()
        # build boolean mask
        failure_mask = np.less(terminal_wealth, start * floor)
        n_failures = failure_mask.sum()
        p_fail = n_failures / n_scenarios
        # this is the equivalent of CVaR. When you go below the floor which is the average shortfall. This is the
        # conditional mean of all the outcomes that end up below the floor
        # np.dot is the dot product
        e_shortfall = (
            np.dot(terminal_wealth - start * floor, failure_mask) / n_failures
            if n_failures > 0
            else 0.0
        )

        # Plot!
        fig, (wealth_ax, hist_ax) = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            gridspec_kw={"width_ratios": [3, 2]},
            figsize=(24, 9),
        )
        plt.subplots_adjust(wspace=0.0)
        wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
        wealth_ax.axhline(y=start, ls=":", color="black")
        wealth_ax.axhline(y=start * floor, ls="--", color="red")
        wealth_ax.set_xlim(xmin=0, xmax=120)
        wealth_ax.set_ylim(top=y_max)

        terminal_wealth.plot.hist(
            ax=hist_ax, bins=50, ec="w", fc="indianred", orientation="horizontal"
        )
        hist_ax.axhline(y=start, ls=":", color="black")
        hist_ax.axhline(y=tw_mean, ls=":", color="blue")
        hist_ax.axhline(y=tw_median, ls=":", color="purple")
        hist_ax.annotate(
            f"Mean: ${int(tw_mean)}",
            xy=(0.65, 0.95),
            xycoords="axes fraction",
            fontsize=18,
        )
        hist_ax.annotate(
            f"Median: ${int(tw_median)}",
            xy=(0.65, 0.9),
            xycoords="axes fraction",
            fontsize=18,
        )

        if floor > 0.01:
            hist_ax.axhline(y=start * floor, ls="--", color="red", linewidth=3)
            hist_ax.annotate(
                f"Violations: {n_failures} {p_fail*100:.2f}%",
                xy=(0.65, 0.85),
                xycoords="axes fraction",
                fontsize=18,
            )
            hist_ax.annotate(
                f"E(shortfall)= ${e_shortfall:2.2f}",
                xy=(0.65, 0.8),
                xycoords="axes fraction",
                fontsize=18,
            )

    def discount(
        self, t: pd.Series, r: float | pd.Series | pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the price of a pure discount bond that pays a dollar at time period t
        and r is the per-period interest rate. It returns a |t| x |r| Series or DataFrame.
        r can be a float, Series or Dataframe returns a DataFrame indexed by t
        """
        discounts = pd.DataFrame([(r + 1) ** -i for i in t])
        discounts.index = t
        return discounts

    def pv(self, flows: pd.Series, r: float):
        """
        Computes the present value of a sequence of liabilities
        given by the time as an index, and amounts. r can be a scalar,
        or a Series or a DataFrame with the number of rows matching the num of rowws in flows
        """
        dates = flows.index
        discounts = self.discount(dates, r)
        return discounts.multiply(flows, axis="rows").sum()

    # Funding Ratio:
    def funding_ratio(
        self, assets: pd.Series, liabilities: pd.Series, r: float
    ) -> float:
        """
        Computes the funding ratio of some assets that are modelled as a series of cash flows
        given liabilities and interest rates
        """
        return self.pv(assets, r) / self.pv(liabilities, r)

    def inst_to_ann(self, r: float) -> float:
        """
        Convert short rate to annualized rate
        """
        return np.expm1(r)

    def ann_to_inst(self, r: float) -> float:
        """
        Convert annualized to short rate
        """
        return np.log1p(r)

    def cir(
        self,
        n_years=10,
        n_scenarios=1,
        a=0.05,
        b=0.03,
        sigma=0.05,
        steps_per_year=12,
        r_0=None,
    ):
        """
        Generate random interest rate evolution over time using the CIR model
        b and r_0 are assumed to be the annualized rates, not the short rate
        and the returned values are the annualized rates as well
        """
        if r_0 is None:
            r_0 = b
        r_0 = self.ann_to_inst(r_0)
        dt = 1 / steps_per_year
        num_steps = (
            int(n_years * steps_per_year) + 1
        )  # because n_years might be a float

        shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(shock)
        rates[0] = r_0

        ## For Price Generation
        h = math.sqrt(a**2 + 2 * sigma**2)
        prices = np.empty_like(shock)
        ####

        def price(ttm, r):
            _A = (
                (2 * h * math.exp((h + a) * ttm / 2))
                / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
            ) ** (2 * a * b / sigma**2)
            _B = (2 * (math.exp(h * ttm) - 1)) / (
                2 * h + (h + a) * (math.exp(h * ttm) - 1)
            )
            _P = _A * np.exp(-_B * r)
            return _P

        prices[0] = price(n_years, r_0)
        ####

        for step in range(1, num_steps):
            r_t = rates[step - 1]
            d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
            rates[step] = abs(r_t + d_r_t)
            # generate prices at time t as well ...
            prices[step] = price(n_years - step * dt, rates[step])

        rates = pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))
        ### for prices
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        ###
        return rates, prices

    def bond_cash_flows(
        self,
        maturity: float,
        principal: float = 100,
        coupon_rate: float = 0.03,
        coupons_per_year: int = 12,
    ) -> pd.Series:
        """
        Returns a series of cash flows generated by a bond, indexed by a coupon number
        principal is also called the face value of the bond
        """
        # number of total coupons to be paid
        n_coupons = round(maturity * coupons_per_year)
        coupon_amt = principal * coupon_rate / coupons_per_year
        coupon_times = np.arange(1, n_coupons + 1)
        cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
        cash_flows.iloc[-1] += principal
        return cash_flows

    def bond_price(
        self,
        maturity: float,
        principal: float = 100,
        coupon_rate: float = 0.03,
        coupons_per_year: int = 12,
        discount_rate: float | pd.DataFrame = 0.03,
    ):
        """
        Computes the price of a bond that pays regular coupons until maturity
        at which time the principal and the final coupon is returned
        This is not designed to be efficient, rather, it is to illustrate
        the underlying principle behind bond pricing!
        If discount_rate is a DataFrame, then it is assumed to be the rate on each coupon date
        and the bond value is computed over time.
        i.e. the index od the discount_rate DataFrame is assumed to be the copuon number
        """
        if isinstance(discount_rate, pd.DataFrame):
            pricing_dates = discount_rate.index
            prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
            for t in pricing_dates:
                prices.loc[t] = self.bond_price(
                    maturity - t / coupons_per_year,
                    principal,
                    coupon_rate,
                    coupons_per_year,
                    discount_rate.loc[t],
                )
            return prices

        # base case... single time period
        else:
            if maturity <= 0:
                return principal + principal * coupon_rate / coupons_per_year
            cash_flows = self.bond_cash_flows(
                maturity, principal, coupon_rate, coupons_per_year
            )
            return self.pv(cash_flows, discount_rate / coupons_per_year)

    def bond_total_return(
        self,
        monthly_prices: pd.DataFrame,
        principal: float,
        coupon_rate: float,
        coupons_per_year: int,
    ):
        """
        Computes the total return of a bond based on monthly bond prices and coupon payments.
        Assumes that dividens (coupons) are paid out at the end of the period
        (e.g. end of 3 months for quarterly dividends) and that dividends
        are reinvested in the bonds
        """
        coupons = pd.DataFrame(
            data=0, index=monthly_prices.index, columns=monthly_prices.columns
        )
        t_max = monthly_prices.index.max()
        pay_date = np.linspace(
            12 / coupons_per_year, t_max, int(coupons_per_year * t_max / 12), dtype=int
        )
        coupons.iloc[pay_date] = principal * coupon_rate / coupons_per_year
        total_return = (monthly_prices + coupons) / monthly_prices.shift() - 1
        return total_return.dropna()

    def macaulay_duration(self, flows: pd.Series, discount_rate: float) -> float:
        """
        Computes the Macaulay Duration of a sequence of cash flows
        """
        discounted_flows = self.discount(flows.index, discount_rate) * flows
        weights = discounted_flows / discounted_flows.sum()
        return np.average(flows.index, weights=weights)
    
    def macaulay_duration2(self, flows, discount_rate):
        discounted_flows = self.discount(flows.index, discount_rate)*pd.DataFrame(flows)
        weights = discounted_flows/discounted_flows.sum()
        return np.average(flows.index, weights=weights.iloc[:,0])

    def match_duration(
        self, cf_t: pd.Series, cf_s: pd.Series, cf_l: pd.Series, discount_rate: float
    ) -> float:
        """
        Returns the weight w in cf_s (short bond) that, along with (1-w) in cf_l
        will have an effect duration that matches cf_t (target)
        """
        d_t = self.macaulay_duration(cf_t, discount_rate)
        d_s = self.macaulay_duration(cf_s, discount_rate)
        d_l = self.macaulay_duration(cf_l, discount_rate)

        return (d_l - d_t) / (d_l - d_s)
    
    def bt_mix(self, r1: pd.DataFrame, r2: pd.DataFrame, allocator, **kwargs) -> pd.DataFrame:
        """
        Given to sets of return and an allocator, create a portfolio that mix between the two over time.
        Runs a backtest (simulation) of allocating between two sets of returns.
        r1 and r2 are TxN dataframes where T is the time step and N is the numbero of scenarios.
        Allocator is a function that takes two sets of returns and allocator specific parameters,
        and produces an allocation to the first portfolio (the rest of the money is invested
        in GHP) as a Tx1 dataframe.
        Returns a TxN dataframe of the resulting N portfolio scenarios
        """
        if not r1.shape == r2.shape:
            raise ValueError("r1 and r2 need to be to same shape")
            
        weights = allocator(r1, r2, **kwargs)
        
        if not weights.shape == r1.shape:
            raise ValueError("Allocator returned weights that don't match r1")
        
        return weights*r1 + (1-weights)*r2
    
    def terminal_values(self, rets: pd.Series) -> float:
        """
        Returns the final values of a dolalr at the end of each period for each scenario
        """
        return (rets+1).prod()
    
    
    def terminal_stats(self, rets: pd.DataFrame,floor= 0.8, cap=np.inf, name="Stats") -> pd.DataFrame:
        """
        Produces Summary Statistics on the terminal values per invested dollar across
        a range of N scenarios.
        Rets is a TxN dataframe of returns, where T is the time step (we assume rets is sorted by tine)
        Returns a 1 column dataframe of summary stats indexed by the stat name
        params:
        floor: is the minimum essential goal we want to achieve
        """
        terminal_wealth = (rets+1).prod()
        breach = terminal_wealth < floor
        reach = terminal_wealth >= cap
        p_breach = breach.mean() if breach.sum()>0 else np.nan
        p_reach = reach.mean() if reach.sum()>0 else np.nan
        e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
        e_surplus = (-cap + terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
        
        sum_stats = pd.DataFrame.from_dict({
                            "mean": terminal_wealth.mean(),
                            "std" : terminal_wealth.std(),
                            "p_breach": p_breach,
                            "e_short": e_short,
                            "p_reach": p_reach,
                            "e_surplus": e_surplus
                            }, orient="index", columns = [name])
        
        return sum_stats
        
        

    
#################      ALLOCATORS     ############################    
    
def fixedmix_allocator(r1: pd.DataFrame, r2: pd.DataFrame, w1: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Produces a time series over T steps of allocation between PSP and GHP across N scenarios
    PSP and GHP are TxN dataframes that represent the returns of the PSP and GHP such that:
    each column is a scenario
    each row is the price for a timestep
    Returns a TxN dataframe of PSP weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def glidepath_allocator(r1: pd.DataFrame, r2: pd.DataFrame, start_glide: float =1, end_glide: float =0) -> pd.DataFrame:
    """
    simulate a Target-Date Fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data= np.linspace(start_glide, end_glide, num= n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r: pd.DataFrame, ghp_r: pd.DataFrame, floor: float, zc_prices: pd.DataFrame, m: int=3) -> pd.DataFrame:
    """
    Allocates between PSP and GHP with the goal to provide exposure to the upside of PSP
    without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PHP.
    Returns a dataframe  with the same shape as the PSP/GHP representing the weights
    in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value =  np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        # PV of floor assuming today's rates and flat YC. ZC_prices are need to discount the  floor price
        floor_value = floor*zc_prices.iloc[step]
        cushion = (account_value -  floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    
    return w_history

def drawdown_allocator(psp_r: pd.DataFrame, ghp_r: pd.DataFrame, maxdd: float, m: int=3) -> pd.DataFrame:
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value =  np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### floor is based on previous peak
        cushion = (account_value -  floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value =np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
        
    return w_history
        
        
        

##############################  Not used anymore ########################################
def plot_ef2(metrics, n_points: int, returns: pd.Series, cov: pd.DataFrame):
    """
    Plots the efficient frontier of a two-portfolio asset
    """
    if returns.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("function can only plot 2-asset frontier")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [metrics.portfolio_return(w, returns) for w in weights]
    vols = [metrics.portfolio_vol(w, cov) for w in weights]
    eff_frontier = pd.DataFrame({"R": rets, "Vol": vols})
    eff_frontier.plot.line(x="Vol", y="R", style=".-")


def show_cppi(
    n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0.0, riskfree_rate=0.03, y_max=100
) -> None:
    """
    Plot the results of a Monte carlo simulation of CPPI
    """
    start = 100
    sim_rets = self.gbm(
        n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12
    )
    risky_rets = pd.DataFrame(sim_rets)
    # run the backtest
    btr = self.run_cppi(
        risky_r=pd.DataFrame(risky_rets),
        riskfree_rate=riskfree_rate,
        m=m,
        start=start,
        floor=floor,
    )
    wealth = btr["Wealth"]
    y_max = (
        wealth.values.max() * y_max / 100
    )  # scale max value: effect is to zoom in/out a portion of the plot
    ax = wealth.plot(legend=False, alpha=0.3, color="indianred", figsize=(12, 6))
    ax.axhline(y=start, ls=":", color="black")
    ax.axhline(y=start * floor, ls="--", color="red")
    ax.set_ylim(top=y_max)

    def cir(
        self,
        n_years=10,
        n_scenarios=1,
        a=0.05,
        b=0.03,
        sigma=0.05,
        steps_per_year=12,
        r_0=None,
    ) -> pd.DataFrame:
        """
        Implements the CIR model for interest rates
        """
        if r_0 is None:
            r_0 = b

        r_0 = self.ann_to_inst(r_0)
        dt = 1 / steps_per_year

        num_steps = int(n_years * steps_per_year) + 1
        # generating the shock
        dWt = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(dWt)
        rates[0] = r_0

        for step in range(1, num_steps):
            r_t = rates[step - 1]
            d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * dWt[step]
            rates[step] = abs(
                r_t + d_r_t
            )  # abs is to make sure that it is always positive (it might become negative due to high shock)

        return pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))


def pv(self, flows: pd.Series, r: float) -> float:
    """
    Computes the present value of a sequence of liabilities
    flows is index by the time, and the values are the amounts of each liability
    returns the present value of the sequence
    """
    dates = flows.index
    discounts = self.discount(dates, r)
    return (discounts * flows).sum()


def discount(self, t: pd.Series, r: float) -> float:
    """
    Computes the price of a pure discount bond (ZCB) thta pays 1 dollar at time t,
    given the interest rate r
    """
    return 1 / (1 + r) ** t


def bond_price(
    self,
    maturity: float,
    principal: float = 100,
    coupon_rate: float = 0.03,
    coupons_per_year: int = 12,
    discount_rate: float = 0.03,
) -> float:
    """
    Price a bond based on bond params: maturity, principal, coupon rate, coupons_per_year and
    the prevailing discount_rate discount_rate is the current interest rate
    """
    cash_flows = self.bond_cash_flows(
        maturity, principal, coupon_rate, coupons_per_year
    )
    return self.pv(cash_flows, discount_rate / coupons_per_year)
