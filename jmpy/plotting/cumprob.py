import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def cumprob(x,
            data: pd.DataFrame=None,
            legend=None,
            figsize: tuple=(9, 6),
            xscale: str='linear',
            yscale: str='linear',
            cmap: str='default',
            alpha: float=0.5,
            marker: str='.',
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
    :param marker: set matplotlib marker
    :param table: bool, default is True, prints the datatable summary to the graph
    :param kwargs:  passed to matplotlib hist function
    :param fig: matplotlib figure if you want to reuse the figure.
    :return: matplotlib figure
    """

    # if no dataframe is supplied, create one
    if data is None:
        x, _, legend, data = components.create_df(x, None, legend)

    local_data = data.copy()
    local_data = local_data.reset_index()
    local_data[x] = local_data[x].astype('float').dropna()

    min_, max_ = np.min(local_data[x]), np.max(local_data[x])

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(None, legend, table, fig=fig)

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
        axl.set_title(legend,loc='left')

        for group in set(local_data[legend]):
            axm = components.cumprob(local_data[local_data[legend] == group][x],
                                       axm,
                                       color=legend_color[group],
                                       marker=marker,
                                       alpha=alpha)
    else:
        axm = components.cumprob(local_data[x], axm, marker=marker, alpha=alpha)

    # various formating
    for label in axm.get_xticklabels():
        label.set_rotation(90)
    axm.set_xlim(min_, max_)
    axm.set_xscale(xscale)
    axm.set_yscale(yscale)
    axm.set_xlabel(x)

    return canvas.figure
