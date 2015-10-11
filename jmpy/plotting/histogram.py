import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def histogram(x, data=None, legend=None, figsize=(12, 6),
              xscale='linear', yscale='linear', cmap='default',
              alpha=0.5, cumprob=False, marker='.', bins=25,
              table=True, fig=None, axes=None, cgrid=None, **kwargs):
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
        (x, _, _, legend, _, _), data = components.create_df(x, None, legend)

    df = data.copy()
    df = df.reset_index()
    df[x] = df[x].astype('float').dropna()

    min_, max_ = np.min(df[x]), np.max(df[x])

    binlist = np.linspace(min_, max_, bins)

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    elif axes:
        axm = axes
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(cumprob, legend, table, fig=fig)

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
            axl.set_title(legend, loc='left')

        for group in sorted(set(df[legend])):
            axm.hist(np.asarray(df[df[legend] == group][x]),
                     alpha=alpha,
                     bins=binlist,
                     color=legend_color[group],
                     label=str(group),
                     **kwargs)
            if cumprob and not axes:
                axc = components.cumprob(df[df[legend] == group][x],
                                         axc,
                                         color=legend_color[group],
                                         marker=marker,
                                         alpha=alpha)
    else:
        axm.hist(np.asarray(df[x]),
                 alpha=alpha,
                 bins=binlist,
                 **kwargs)
        if cumprob:
            axc = components.cumprob(df[x], axc, marker=marker, alpha=alpha)

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

    if axes:
        return axm

    return canvas.figure
