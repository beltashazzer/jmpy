import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def colormap(series, kind: str='discrete', cmap: str='default'):
    if kind == 'discrete' and cmap != 'default':
        c = plt.get_cmap(cmap)

        # break up the color map into equal portions
        r = np.linspace(0, c.N, len(set(series.dropna())))
        r = r.astype('int')

        # convert the color map to hex string for matplotlib scatter plot color
        cc = [str.upper(mpl.colors.rgb2hex(c(i))) for i in r]
        colors = {i: c for i, c in zip(sorted((set(series))), cc)}

        # map the color grid to the series column
        cgrid = series.apply(lambda x: colors[x])
        return cgrid

    elif kind == 'discrete' and cmap == 'default':
        colors = {i: c for i, c in zip(sorted((set(series))), it.cycle(plt.rcParams['axes.color_cycle']))}
        cgrid = series.apply(lambda x: colors[x])
        return cgrid

    else:
        raise NotImplementedError
