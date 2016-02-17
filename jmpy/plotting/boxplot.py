import collections as col

import numpy as np
import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def boxplot(x, y, data=None, legend=None, marker='o',
            alpha=.5, points=True, cumprob=False, yscale='linear',
            cmap='default', figsize=(12, 6),  orderby=None, table=True,
            fig=None, axes=None, cgrid=None, violin=False, **kwargs):
    """
    Boxplot function
    :param x: str or ndarray
    :param y: str or ndarray
    :param data: pd.Dataframe, source of data
    :param legend: str or ndarray color code by this column
    :param marker: str, default marker to use in plots
    :param alpha: float, alpha for plots
    :param points: bool, display or not display points
    :param cumprob: bool, display cumprob plot?
    :param yscale: str, default = linear, can be log or symlog too
    :param cmap: str, matplotlib colormap
    :param figsize: tuple(int,int), figure size
    :param orderby: str, order x axis by this param
    :param datatable: bool, show or not show datatable is available
    :param fig: matplotlib figure, if you want to re-use the figure, pass in one already created
    :param axes: matplotlib axes, if this is specified, the boxplot will be created on that axes,
                    and other axes will not be created.
    :param kwargs:


    :return: matplotlib figure
    """

    # if no dataframe is supplied, create one
    if data is None:
        (x, y, _, legend, _, _), data = components.create_df(x, y, legend)

    df = data.copy()
    df = df.reset_index()
    df[x] = df[x].astype('str')
    df[y] = df[y].astype('float').dropna()

    # TODO:  this doesn't really work right
    if orderby:
        temp = df.sort(x)
        t = temp.groupby(x)[orderby]
        map_of_x = col.OrderedDict()
        for mg in sorted(t.groups):
            g = t.get_group(mg).reset_index()
            map_of_x[mg] = g[orderby][0]

        list_to_order = sorted([value for value in map_of_x.values()])

        order = []
        x_to_loc = {}
        for k, v in map_of_x.items():
            idx = list_to_order.index(v)
            x_to_loc[k] = idx
            order.append(idx)

    min_, max_ = np.min(df[y]), np.max(df[y])

    # if an axis is supplied, we will not create another one
    # if a figure is supplied, we will reuse the figure
    if fig and not axes:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    elif axes:
        axm = axes
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(cumprob, legend, table, fig=fig)

    if violin:
        array = []
        for arr in sorted(set(df[x])):
            array.append(df[df[x] == arr][y])

        axm.violinplot(array, showmedians=True)

    else:
        if orderby:
            df.boxplot(column=y, by=x, ax=axm, showfliers=False,
                       positions=order, fontsize=8, **kwargs)
        else:
            df.boxplot(column=y, by=x, ax=axm, showfliers=False, fontsize=8, **kwargs)

    # We need to identify all of the unique entries in the groupby column
    unique_groups = set(df[x])
    nonan_grps = []
    for group in unique_groups:
        if 'nan' not in group:
            nonan_grps.append(group)

    if legend:
        # colormap is supposed to be the goto function to get all colormaps
        # should return a colorgrid that maps each point to a set of colors
        # if cgrid is already supplied, we will re-use that color grid
        if cgrid is None:
            cgrid = common.colors.colormap(df[legend], kind='discrete', cmap=cmap)

        legend_color = {}
        for i, key in df[legend].iteritems():
            legend_color[key] = cgrid[i]

        if not axes:  # skip over creation of legend if axes is provided
            axl = components.legend(sorted(list(legend_color.items())), axl)
            axl.set_title(legend, loc='left')

    # add all the point level data
    groups = sorted(nonan_grps)
    for j, val in enumerate(groups):
        ys = df[y][df[x] == val]
        if orderby:
            pos = x_to_loc[val]
            xs = np.random.normal(pos, 0.05, size=len(ys))
        else:
            # create the jitters for the points
            xs = np.random.normal(j + 1, 0.05, size=len(ys))

        if points:

            # if cgrid is None, that is the standard way of creating the plot
            # cgrid is typically supplied by the jp.grid function
            if legend or cgrid is not None:
                cs = cgrid[df[x] == val]
                axm.scatter(xs, ys.values, color=cs, marker=marker, alpha=alpha,
                            linewidths=1, **kwargs)
            else:
                axm.scatter(xs, ys.values, marker=marker, alpha=alpha,
                            linewidths=1, **kwargs)

        # skip creating the cumprob plot if the axes was supplied
        if cumprob and not axes:
            if legend:
                cs = cgrid[df[x] == val]
                axc = components.cumprob(ys, axc, color=cs, alpha=alpha, swapxy=True)
            else:
                axc = components.cumprob(ys, axc, alpha=alpha, swapxy=True)

    # various formating
    axm.set_ylim(min_, max_)
    axm.set_yscale(yscale)
    axm.set_ylabel(y)
    for label in axm.get_xticklabels():
        label.set_rotation(90)

    if cumprob and not axes:
        axc.set_ylim(min_, max_)
        axc.set_yscale(yscale)
        axc.set_yticklabels([], visible=False)

        for label in axc.get_xticklabels():
            label.set_rotation(90)

    if table and not axes:
        components.datatable(y, data, axt, by=x)

    axm.set_title('')

    if axes:
        return axm

    fig.suptitle('')
    return canvas.figure
