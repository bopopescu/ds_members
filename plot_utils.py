"""
Some helper functions to plot
"""

import pandas as pd
import numpy as np

def get_quantile(data, var, q=.9):
    """Return only the (10 x q)% of entries along the var dimension.
    
    Arguments:
        data {pd.DataFrame} -- 
        var {str} -- column to calculate quantile
    
    Keyword Arguments:
        q {float} -- quantile (default: {.9})
    
    Returns:
        pd.DataFrame -- Portion of the data frame up to q
    """
    h = data.sort_values(by=var, ascending=True)
    tot = h.shape[0]
    q_idx = int(np.ceil(q * tot))

    h_lo = h.iloc[:q_idx, :]

    return h_lo


def binned_df(data, var, min=None, max=None, n=20):
    """Bin a dataframe along an axis.
    
    Arguments:
        data {pd.DataFrame} -- 
        var {str} -- column to bin on
    
    Returns:
        pd.DataFrama -- binned data frame
    """
    m = min if min else data[var].min()
    M = max if max else data[var].max()
    bin_size = (M - m) / n 

    # bin_size = (data[var].max() - data[var].min()) / 20
    # bins_range = np.arange(data[var].min(), data[var].max() + bin_size,
    #                        bin_size)
    bins_range = np.arange(m, M + bin_size, bin_size)

    binned = pd.cut(data.loc[:, var], bins_range)
    binned.dropna(inplace=True)

    h = pd.value_counts(binned)
    h = pd.DataFrame(h)
    h[str(var) + '_mid'] = [p.mid for p in h.index]

    return h
