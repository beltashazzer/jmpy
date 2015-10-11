import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def cumprob(x, data=None, legend=None, figsize=(12, 6),
            xscale='linear', yscale='linear', cmap='default', alpha=0.5,
            marker='.', table=True, fig=None, axes=None, cgrid=None, **kwargs):
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
        (x, _, _, legend, _, _), data = components.create_df(x, None, legend)

    df = data.copy()
    df = df.reset_index()
    df[x] = df[x].astype('float').dropna()

    min_, max_ = np.min(df[x]), np.max(df[x])

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    elif axes:
        axm = axes
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(None, legend, table, fig=fig)

    if table and not axes:
        axt = components.datatable(x, data, axt, by=legend)

    if legend:
        # colormap is supposed to be the goto function to get all colormaps
        # should return a colorgrid that maps each point to a set of colors
        if cgrid is None:
            cgrid = common.colors.colormap(df[legend], kind='discrete', cmap=cmap)


        legend_color = {}
        for i, key in df[legend].iteritems():
            legend_color[key] = cgrid[i]

        if not axes:
            axl = components.legend(sorted(list(legend_color.items())), axl)
            axl.set_title(legend,loc='left')

        for group in sorted(set(df[legend])):
            axm = components.cumprob(df[df[legend] == group][x],
                                       axm,
                                       color=legend_color[group],
                                       marker=marker,
                                       alpha=alpha)
    else:
        axm = components.cumprob(df[x], axm, marker=marker, alpha=alpha)

    # various formating
    for label in axm.get_xticklabels():
        label.set_rotation(90)
    axm.set_xlim(min_, max_)
    axm.set_xscale(xscale)
    axm.set_yscale(yscale)
    axm.set_xlabel(x)

    if axes:
        return axm

    return canvas.figure
