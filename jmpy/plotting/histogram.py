import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def histogram(x,
              data: pd.DataFrame=None,
              legend=None,
              figsize: tuple=(12, 6),
              xscale: str='linear',
              yscale: str='linear',
              cmap: str='default',
              alpha: float=0.5,
              cumprob: bool=False,
              marker: str='.',
              bins=25,
              table=True,
              fig=None,
              **kwargs):
    """
    :param x:  str or ndarray
    :param data: is x is a str, this is a pd.Dataframe
    :param legend: str or ndarray,
    :param figsize: default is 9,6; sets the figure size
    :param xscale: default is linear, set the scale type [linear, log, symlog]
    :param yscale: default is linear, set the scale type [linear, log, symlog]
    :param cmap: colormap to use for plotting
    :param alpha: default is 0.5
    :param cumprob: bool, determines if cumprob plot is displayed
    :param marker: set matplotlib marker
    :param bins: # of bins to use
    :param table: bool, default is True, prints the datatable summary to the graph
    :param kwargs:  passed to matplotlib hist function
    :param fig: matplotlib figure instance for re-use...
    :return:
    """

    # if no dataframe is supplied, create one
    if data is None:
        x, _, legend, data = components.create_df(x, None, legend)

    local_data = data.copy()
    local_data = local_data.reset_index()
    local_data[x] = local_data[x].astype('float').dropna()

    min_, max_ = np.min(local_data[x]), np.max(local_data[x])

    binlist = np.linspace(min_, max_, bins)

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(cumprob, legend, table, fig=fig)

    if table:
        axt = components.datatable(x, data, axt, by=legend)

    if legend:
        # colormap is supposed to be the goto function to get all colormaps
        # should return a colorgrid that maps each point to a set of colors
        cgrid = common.colors.colormap(local_data[legend], kind='discrete', cmap=cmap)

        legend_color = {}
        for i, key in local_data[legend].iteritems():
            legend_color[key] = cgrid[i]

        axl = components.legend(sorted(list(legend_color.items())), axl)
        axl.set_title(legend, loc='left')

        for group in set(local_data[legend]):
            axm.hist(np.asarray(local_data[local_data[legend] == group][x]),
                     alpha=alpha,
                     bins=binlist,
                     color=legend_color[group],
                     **kwargs)
            if cumprob:
                axc = components.cumprob(local_data[local_data[legend] == group][x],
                                         axc,
                                         color=legend_color[group],
                                         marker=marker,
                                         alpha=alpha)
    else:
        axm.hist(np.asarray(local_data[x]),
                 alpha=alpha,
                 bins=binlist,
                 **kwargs)
        if cumprob:
            axc = components.cumprob(local_data[x], axc, marker=marker, alpha=alpha)

    # various formating
    axm.set_xlim(min_, max_)
    axm.set_xscale(xscale)
    axm.set_yscale(yscale)
    axm.set_xlabel(x)

    for label in axm.get_xticklabels():
        label.set_rotation(90)

    if cumprob:
        axc.set_xlim(min_, max_)
        axc.set_xscale(xscale)
        axc.set_yticklabels([], visible=False)
        for label in axc.get_xticklabels():
            label.set_rotation(90)

    return canvas.figure
