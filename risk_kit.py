import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.6f}'.format
plt.rcParams["figure.figsize"] = (10,6)


def pre_processing_hfi():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    # some Pre-Processing
    hfi = hfi/100
    hfi.index = hfi.index.to_period("M")
    return hfi

def pre_processing_ind():
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",  header = 0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind30_size():
    """
    """
    ind = pd.read_csv("data/ind30_m_size.csv",  header = 0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind30_nfirms():
    """
    Loads and format the Ken French 30 Industry Portfolios Value weighted monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv",  header = 0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    # fix column name by removing spaces
    ind.columns = ind.columns.str.strip()
    return ind

class Metrics():
    """
    This class computes common metrics from a time series of asset returns
    with these data
    """
    def __init__(self, starting_amount: int=1):
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
        return (weights.T @ covmat @ weights)**0.5
    
    def annualized_rets(self, series: pd.Series, periods_per_year = 12):
        """
        Annualize the returns of a series by inferring the periods per year
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """
        compounded_growth = (1+series).prod()
        n_periods = series.shape[0]
        return compounded_growth**(periods_per_year/n_periods) - 1
                        
    
    def annualized_vol(self, series: pd.Series, periods_per_year = 12):
        """
        Annualize the volatility of a series by inferring the periods per year
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """
        return series.std()*(periods_per_year**0.5)
    
    def sharpe_ratio(self, series: pd.Series, riskfree_rate = 0.03, periods_per_year = 12):
        """
        Computes the annualized Sharpe Ratio of a series
        PARAMETERS:
        periods_per_year = 12 for monthly data, 252 for stock daily data, ecc...
        """
                     
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year) - 1
        excess_ret = series - rf_per_period
        ann_ex_ret = self.annualized_rets(excess_ret, periods_per_year)
        ann_vol = self.annualized_vol(series, periods_per_year)
        return ann_ex_ret/ann_vol

    def skewness(self, series: pd.Series):
        """
        Compute the skewness of the supplied Series/Dataframe
        """
        demeaned_data = series - series.mean()
        # population std dddof=0
        sigma_data = series.std(ddof=0)
        exp =  (demeaned_data**3).mean()
        return exp/sigma_data**3
    
    def kurtosis(self, series: pd.Series):
        """
        Compute the kurtosis of the supplied Series/Dataframe
        """
        demeaned_data = series - series.mean()
        # population std dddof=0
        sigma_data = series.std(ddof=0)
        exp =  (demeaned_data**4).mean()
        return exp/sigma_data**4
    
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
        excess_negative = excess[excess<0]
        excess_negative_square = excess_negative**2
        n_negative = (excess<0).sum()
        return (excess_negative_square.sum()/n_negative)**0.5
    
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
        z_score = norm.ppf(level/100)
        if modified:
            s = self.skewness(series)
            k = self.kurtosis(series)
            z_score = (z_score + 
                (z_score**2 - 1)*s/6 +
                (z_score**3 - 3*z_score)*(k-3)/24 -
                (2*z_score**3 - 5*z_score)*(s**2)/36
                )
                
        return -(series.mean() + z_score * series.std(ddof=0))
    
    def historical_cvar(self, series: pd.Series):
        """
        computes CVaR based based on historical data over the data frequency.
        """
        is_beyond = series <= - self.historical_var(series)
        
        return -series[is_beyond].mean()
    

    def drawdown(self, data: pd.Series) -> pd.DataFrame:
        """
        Computes the drawdown of a time series
        """
        #fig, axs = plt.subplots(2)
        # compute wealth index
        wealth_index= self.starting_amount*(1 + data).cumprod()
        #print(wealth_index)
        #axs[0].plot(wealth_index.index.to_timestamp(), wealth_index)
        # compute previuos peaks
        previuos_peaks = wealth_index.cummax()
        #axs[0].plot(previuos_peaks.index.to_timestamp(), previuos_peaks)
        ## compute drawdown
        drawdowns = (wealth_index - previuos_peaks)/previuos_peaks
        #axs[1].plot(drawdowns.index.to_timestamp(), drawdowns)
        #print(f"max drawdown is {drawdowns.min()*100:.2f}% located at {drawdowns.idxmin()}\n")

        return pd.DataFrame({"Wealth": wealth_index, "Peaks": previuos_peaks, "Drawdown": drawdowns})
    
    
    def minimize_vol(self, target_return: float, rets: pd.Series, cov: pd.DataFrame):
        """
        Minimizes volatility for a given target return
        subject to the constraints of the Mean-Variance Portfolio
        """
        n_assets = rets.shape[0]

        # initial guess
        init_guess = np.repeat(1/n_assets, n_assets)

        # no shorting (it's a tuple of tuples)
        bounds = ((0, 1.0),)*n_assets

        # for a given target return the portfolio return is equal
        return_is_target = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda weights, rets:  target_return - self.portfolio_return(weights, rets)
        }

        # weights su to 1
        weights_sum_to_one = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) -1
        }

        results = minimize(self.portfolio_vol,
                           init_guess,
                           args = (cov,),
                           method= "SLSQP",
                           options = {"disp": False},
                           constraints = (return_is_target, weights_sum_to_one),
                           bounds = bounds
                          )
        return results.x
    
    
    def _optimal_weights(self, n_points: int, rets: pd.Series, cov: pd.DataFrame) -> list[np.array]:
        """
        list of weights to run the optimizer on to minimize volatility
        for a given target return
        
        RiskFree rate  + ER + COV -> weights
        """
        # generate target return between minimun and maximum target returns
        target_rs = np.linspace(rets.min(), rets.max(), n_points)
        # find the weights that minimizes volatility given a certain target return
        weights = [self.minimize_vol(target_return, rets, cov) for target_return in target_rs]
        return weights
    
    def _neg_sharpe_ratio(self, weights: pd.Series, riskfree_rate: float, rets: pd.Series, cov: pd.DataFrame):
        """
        Returns the negative of the Sharpe Ratio
        """
        r = self.portfolio_return(weights, rets)
        vol = self.portfolio_vol(weights, cov)
        
        return -((r - riskfree_rate)/vol)
    
    def maximum_sharpe_ratio(self, riskfree_rate: float, rets: pd.Series, cov: pd.DataFrame):
        """
        Computes the Maximum Sharpe Ratio Portfolio
        subject to the constraints of the Mean-Variance Portfolio
        """
        n_assets = rets.shape[0]
        # initial guess
        init_guess = np.repeat(1/n_assets, n_assets)
        # no shorting (it's a tuple of tuples)
        bounds = ((0, 1.0),)*n_assets
        # weights su to 1
        weights_sum_to_one = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) -1
        }

        results = minimize(self._neg_sharpe_ratio,
                           init_guess,
                           args = (riskfree_rate, rets, cov,),
                           method= "SLSQP",
                           options = {"disp": False},
                           constraints = (weights_sum_to_one),
                           bounds = bounds
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
        return self.maximum_sharpe_ratio(riskfree_rate = rfr, rets = returns, cov= cov)

    def plot_ef(self, n_points: int, returns: pd.Series, cov: pd.DataFrame, \
                show_cml = False, riskfree_rate: float = 0., show_ew = False, show_gmv = False) -> None:
        """
        Plots the multi-asset efficient frontier
        """
        weights = self._optimal_weights(n_points, returns, cov)
        #print(weights)
        rets = [self.portfolio_return(w, returns) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        eff_frontier = pd.DataFrame({"R": rets, "Vol": vols})
        ax = eff_frontier.plot.line(x="Vol",y="R", style=".-");
        
        if show_ew:
            n_assets = returns.shape[0]
            w_ew = np.repeat(1/n_assets, n_assets)
            r_ew = self.portfolio_return(w_ew, returns)
            vol_ew = self.portfolio_vol(w_ew, cov)
            # display EW (Equal Weighted) Portfolio
            ax.annotate("EW", (vol_ew, r_ew))
            ax.plot([vol_ew], [r_ew], color = "goldenrod", marker = "o", markersize =10)
        
        if show_gmv:
            w_gmv = self.gmv(cov)
            r_gmv = self.portfolio_return(w_gmv, returns)
            vol_gmv = self.portfolio_vol(w_gmv, cov)
            # display GMV (Global minimum Variance) Portfolio
            ax.annotate("GMV", (vol_gmv, r_gmv))
            ax.plot([vol_gmv], [r_gmv], color = "midnightblue", marker = "o", markersize =10)
        
        if show_cml:
            # Add CML plot
            ax.set_xlim(left=0)
            riskfree_rate = riskfree_rate
            w_msr = self.maximum_sharpe_ratio(riskfree_rate, returns, cov)
            r_msr = self.portfolio_return(w_msr, returns)
            vol_msr = self.portfolio_vol(w_msr, cov)
            # display CML (Capital Market Line) Portfolio
            cml_x= [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.annotate("MSR", (vol_msr, r_msr))
            ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed", markersize =10)
        
        return ax
    
    
    def run_cppi(self, risky_r: pd.Series | pd.DataFrame, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown_constraint= None):
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
        floor_value = start*floor
        peak = start # need for drawdown constraint
        
        if isinstance(risky_r, pd.Series):
            risky_r = pd.DataFrame(risky_r, columns=["R"])
        
        if safe_r is None:
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r[:] = riskfree_rate/12
        
        # setup some dataframes to save intermediate results
        account_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)
        
        # CPPI algorithm implementation
        for step in range(n_steps):
            # drawdown constraint
            if drawdown_constraint is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak*(1-drawdown_constraint)
            cushion = (account_value -floor_value)/account_value
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
            safe_alloc  = account_value * safe_w

            # update the account value at the end of the step
            account_value = risky_alloc * (1 + risky_r.iloc[step]) +\
                            safe_alloc  * (1 + safe_r.iloc[step])

            # save values to look and plot their history
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value
        
        # reference for investing only in risky assets
        risky_wealth = start * (1 + risky_r).cumprod()
        
        backtest_results =  {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth,
            "Risky Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_r": risky_r,
            "safe_r": safe_r
           } 
        
        return backtest_results
    
    def summary_stats(self, data: pd.Series | pd.DataFrame, riskfree_rate = 0.03) -> pd.DataFrame:
        """
        Returns a Dataframe with all the relevant metrics for every asset in the data
        """
        
        def _stats(r: pd.Series, riskfree_rate = 0.03) -> dict:
            """
            Returns a dict that contains aggregated summary stats for returns in the Series
            """
            ann_r = self.annualized_rets(r, periods_per_year=12)
            ann_vol = self.annualized_vol(r, periods_per_year=12)
            ann_sr = self.sharpe_ratio(r, riskfree_rate = riskfree_rate ,periods_per_year=12)
            dd = self.drawdown(r)
            skew = self.skewness(r)
            kurt = self.kurtosis(r)
            cf_var5 = self.cornish_fisher_var(r, modified=True)
            hist_cvar5 = self.historical_cvar(r)
            
            return {"Ann. Return": ann_r,
                    "Ann. Volatility": ann_vol,
                    "Skewness": skew,
                    "Kurtosis": kurt,
                    "Corn-Fisher VaR (5%)": cf_var5,
                    "Hist. CVar (5%)": hist_cvar5,
                    "Sharpe Ratio": ann_sr,
                    "Max Drawdown": dd["Drawdown"].min(),
                    #"Max DD date": dd.idxmin()["Drawdown"]
                   }
        
        # Returns a Series (agg applies the function to every column of the data (Dataframe)
        res = data.agg(_stats).to_frame().to_dict()[0]  # returned Series is converted to dict
        # create metrics for every asset
        idx_list = []
        value_list = []
        for idx, values in res.items():
            idx_list.append(idx)
            value_list.append(values)
            
        return pd.DataFrame(index=idx_list, data=value_list)
    
    def gbm(self, n_years=10, n_scenarios = 1000, mu= 0.07, sigma = 0.15, steps_per_year = 12, s_0 = 100.0, prices=True) -> pd.DataFrame:
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
        dt = 1/steps_per_year
        n_steps = int(n_years*steps_per_year) + 1
        # without discretization error ...
        rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size = (n_steps, n_scenarios))
        rets_plus_1[0] = 1
        # from return to prices
        # prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
        ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
        return ret_val
    
    def show_gbm(self, n_scenarios, mu, sigma) -> None:
        """
        Draw the results of an asset price evolution under the Gemetric Brownian Motion Model
        """
        s_0 = 100
        prices = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
        ax = prices.plot(legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(12,5))
        ax.axhline(y=s_0, ls=":", color="black")
        ax.set_ylim(top=400)
        # draw a dot at the origin
        ax.plot(0, s_0, marker="o", color="darkred", alpha=0.2)    
    
    def show_cppi(self, n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100) -> None:
        """
        Plot the results of a Monte Carlo simulation of CPPI
        """
        start = 100
        sim_rets = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
        risky_rets= pd.DataFrame(sim_rets)
        # run the backtest
        btr= self.run_cppi(risky_r=pd.DataFrame(risky_rets), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
        wealth = btr["Wealth"]
        # scale max value: effect is to zoom in/out a portion of the plot
        y_max = wealth.values.max()*y_max/100 # scale max value: effect is to zoom in/out a portion of the plot
        # calculate terminal wealth stats
        terminal_wealth = wealth.iloc[-1]
        tw_mean = terminal_wealth.mean()
        tw_median = terminal_wealth.median()
        # build boolean mask
        failure_mask = np.less(terminal_wealth, start*floor)
        n_failures = failure_mask.sum()
        p_fail = n_failures/n_scenarios
        # this is the equivalent of CVaR. When you go below the floor which is the average shortfall. This is the
        # conditional mean of all the outcomes that end up below the floor
        # np.dot is the dot product
        e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
        
        # Plot!
        fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios": [3,2]}, figsize=(24,9))
        plt.subplots_adjust(wspace=0.0)
        wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
        wealth_ax.axhline(y=start, ls=":", color="black")
        wealth_ax.axhline(y=start*floor, ls="--", color="red")
        wealth_ax.set_xlim(xmin=0, xmax=120)
        wealth_ax.set_ylim(top=y_max)
        
        terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec="w", fc="indianred", orientation="horizontal")
        hist_ax.axhline(y=start, ls=":", color="black")
        hist_ax.axhline(y=tw_mean, ls=":", color="blue")
        hist_ax.axhline(y=tw_median, ls=":", color="purple")
        hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.65,.95), xycoords="axes fraction", fontsize=18)
        hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.65,.9), xycoords="axes fraction", fontsize=18)
        
        if floor > 0.01:
            hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
            hist_ax.annotate(f"Violations: {n_failures} {p_fail*100:.2f}%",
                             xy=(.65,.85), xycoords="axes fraction", fontsize=18)
            hist_ax.annotate(f"E(shortfall)= ${e_shortfall:2.2f}",
          
                             xy=(.65,.8), xycoords="axes fraction", fontsize=18)
    
    def discount(self, t: float, r: float) -> float:
        """
        Computes the price of a pure discount bond (ZCB) thta pays 1 dollar at time t,
        given the interest rate r
        """
        return 1/(1+r)**t

    def pv(self, l: pd.Series, r: float) -> float:
        """
        Computes the present value of a sequence of liabilities
        l is index by the time, and the values are the amounts of each liability
        returns the present value of the sequence
        """
        dates = l.index
        discounts = self.discount(dates, r)
        return (discounts*l).sum()

    
    # Funding Ratio:
    def funding_ratio(self, assets: float, liabilities: pd.Series, r: float) -> float:
        """
        Computes the funding ratio of some assets given liabilities and interest rates
        """

        
        return assets/self.pv(liabilities, r)
    
    def inst_to_ann(self, r:float) -> float:
        """
        Convert short rate to annualized rate
        """
        return np.expm1(r)

    def ann_to_inst(self, r: float) -> float:
        """
        Convert annualized to short rate
        """
        return np.log1p(r)

    
    def cir(self, n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
        """
        Generate random interest rate evolution over time using the CIR model
        b and r_0 are assumed to be the annualized rates, not the short rate
        and the returned values are the annualized rates as well
        """
        if r_0 is None: r_0 = b 
        r_0 = self.ann_to_inst(r_0)
        dt = 1/steps_per_year
        num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float

        shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(shock)
        rates[0] = r_0

        ## For Price Generation
        h = math.sqrt(a**2 + 2*sigma**2)
        prices = np.empty_like(shock)
        ####

        def price(ttm, r):
            _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
            _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
            _P = _A*np.exp(-_B*r)
            return _P
        prices[0] = price(n_years, r_0)
        ####

        for step in range(1, num_steps):
            r_t = rates[step-1]
            d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
            rates[step] = abs(r_t + d_r_t)
            # generate prices at time t as well ...
            prices[step] = price(n_years-step*dt, rates[step])

        rates = pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))
        ### for prices
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        ###
        return rates, prices

           
    
##############################  Not used anymore ########################################
def plot_ef2(metrics, n_points: int, returns: pd.Series, cov: pd.DataFrame):
    """
    Plots the efficient frontier of a two-portfolio asset
    """
    if returns.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("function can only plot 2-asset frontier")
    weights = [np.array([w, 1-w]) for  w in np.linspace(0, 1, n_points)]
    rets = [metrics.portfolio_return(w, returns) for w in weights]
    vols = [metrics.portfolio_vol(w, cov) for w in weights]
    eff_frontier = pd.DataFrame({"R": rets, "Vol": vols})
    eff_frontier.plot.line(x="Vol",y="R", style=".-");
    

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100) -> None:
    """
    Plot the results of a Monte carlo simulation of CPPI
    """
    start = 100
    sim_rets = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_rets= pd.DataFrame(sim_rets)
    # run the backtest
    btr= self.run_cppi(risky_r=pd.DataFrame(risky_rets), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr["Wealth"]
    y_max = wealth.values.max()*y_max/100 # scale max value: effect is to zoom in/out a portion of the plot
    ax = wealth.plot(legend=False, alpha=0.3, color="indianred", figsize=(12,6))
    ax.axhline(y=start, ls=":", color="black")
    ax.axhline(y=start*floor, ls="--", color="red")
    ax.set_ylim(top=y_max)
    
    def cir(self, n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None) -> pd.DataFrame:
        """
        Implements the CIR model for interest rates
        """
        if r_0 is None:
            r_0 = b

        r_0 = self.ann_to_inst(r_0)
        dt = 1/steps_per_year

        num_steps = int(n_years*steps_per_year) + 1
        # generating the shock
        dWt= np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(dWt)
        rates[0] = r_0

        for step in range(1, num_steps):
            r_t = rates[step-1]
            d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*dWt[step]
            rates[step] = abs(r_t + d_r_t) # abs is to make sure that it is always positive (it might become negative due to high shock)
        
        return pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))
