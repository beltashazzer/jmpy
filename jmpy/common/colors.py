import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def colormap(series, kind='discrete', cmap='default', color_dict=False):
    '''  Create a color map for series data

    Params:
    :param series: array of series data
    :param kind: type of color map, no other kind implemented yet
    :param cmap: specify your own color map
    :param color_dict:  If true, returns a dictionary of series: colors, otherwise
                        return an ndarray of colors for each row
    '''
    if kind == 'discrete' and cmap != 'default':
        c = plt.get_cmap(cmap)

        # break up the color map into equal portions
        r = np.linspace(0, c.N, len(set(series.dropna())))
        r = r.astype('int')

        # convert the color map to hex string for matplotlib scatter plot color
        cc = [str.upper(mpl.colors.rgb2hex(c(i))) for i in r]
        colors = {i: c for i, c in zip(sorted((set(series))), cc)}

        if color_dict:
            return colors

        # map the color grid to the series column
        cgrid = series.apply(lambda x: colors[x])
        return cgrid

    elif kind == 'discrete' and cmap == 'default':
        colors = {i: c for i, c in zip(sorted((set(series))), it.cycle(plt.rcParams['axes.color_cycle']))}
        cgrid = series.apply(lambda x: colors[x])

        if color_dict:
            return colors

        return cgrid

    else:
        raise NotImplementedError
