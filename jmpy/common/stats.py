import collections as col
import numpy as np
import scipy as sp
import scipy.stats as sps


def array_stats(x, data=None, by=None, custom=None):
    """
    Returns a namedtuple with the following attributes:
        mean, stdev, min, ten, med, ninty, max

    TODO:  add custom stat capabilities
    """
    df = data

    stats = col.namedtuple('stats', ['mean', 'stdev', 'min', 'ten', 'med', 'ninty', 'max', 'n'])
    
    # get x, y and by as ndarray's
    if df is not None:
        x = df[x].copy()
        try:
            by = df[by].copy()
            groups = sorted(set(by))
        except ValueError:
            by = None
    
    if by is None:
        return {'All': stats(*_basestats(x))}
    else:
        sdict = col.OrderedDict()
        for g in sorted(groups):
                sdict[g] = stats(*_basestats(x[by == g]))
        return sdict


def _basestats(ndarray):
    """
    Return the mean, stdev,  10%, 50%, 90% of a ndarray
    """

    ndarray = ndarray.copy().dropna()
    
    if ndarray.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean = np.mean(ndarray)
    stdev = np.std(ndarray)
    min_ = np.min(ndarray)
    ten = np.percentile(ndarray, 10)
    med = np.median(ndarray)
    ninty = np.percentile(ndarray, 90)
    max_ = np.max(ndarray)
    n = np.size(ndarray)
    
    return mean, stdev, min_, ten, med, ninty, max_, n

def pct_sigma(array):
    """
    Get normal quantiles

    Parameters
    ----------
    x : array_like
        distribtion of values

    Returns
    -------
    sigma : ndarray
      normal quantile
    pct : ndarray
      percentile
    y : ndarray
      value
    """
    qrank = lambda x: ((x - 0.3175)/(x.max() + 0.365))

    y = array.copy()
    y = y[~np.isnan(y)]
    y.sort()
    
    if y.size == 0:
        blank = np.zeros(y.shape)
        return blank, blank, blank

    n = sp.ones(len(y))
    cs = sp.cumsum(n)
    pct = qrank(cs)
    sigma = sps.norm.ppf(pct)

    return sigma, pct, y
