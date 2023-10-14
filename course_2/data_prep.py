import pandas as pd
import numpy as np

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_fff_returns():
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    rets = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",
                       header=0, index_col=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype is "returns":
        name = f"{weighting}_rets" 
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")
    
    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns(ew=False):
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    def get_ind_file(filetype, ew=False):
        """
        Load and format the Ken French 30 Industry Portfolios files
        """
        known_types = ["returns", "nfirms", "size"]
        if filetype not in known_types:
            raise ValueError(f"filetype must be one of:{','.join(known_types)}")
        if filetype is "returns":
            name = "ew_rets" if ew else "vw_rets"
            divisor = 100
        elif filetype is "nfirms":
            name = "nfirms"
            divisor = 1
        elif filetype is "size":
            name = "size"
            divisor = 1

        ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
        ind.columns = ind.columns.str.strip()
        return ind
    
    return get_ind_file("returns", ew=ew)


def get_ind_returns2(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    
    def get_ind_file(filetype, weighting="vw", n_inds=30):
        """
        Load and format the Ken French Industry Portfolios files
        Variant is a tuple of (weighting, size) where:
            weighting is one of "ew", "vw"
            number of inds is 30 or 49
        """    
        if filetype is "returns":
            name = f"{weighting}_rets" 
            divisor = 100
        elif filetype is "nfirms":
            name = "nfirms"
            divisor = 1
        elif filetype is "size":
            name = "size"
            divisor = 1
        else:
            raise ValueError(f"filetype must be one of: returns, nfirms, size")

        ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
        ind.columns = ind.columns.str.strip()
        return ind
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)

def get_ind_nfirms(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms", n_inds=n_inds)

def get_ind_size(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size", n_inds=n_inds)


def get_ind_market_caps(n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap

def get_total_market_index_returns(n_inds=30):
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_capweight = get_ind_market_caps(n_inds=n_inds)
    ind_return = get_ind_returns(weighting="vw", n_inds=n_inds)
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return