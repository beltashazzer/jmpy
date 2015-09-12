import collections as col

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def boxplot(x,
            y,
            data: pd.DataFrame=None,
            legend=None,
            marker: str='o',
            alpha: float=.5,
            points: bool=True,
            cumprob: bool=False,
            yscale: str='linear',
            cmap: str='default',
            figsize: tuple=(12, 6),
            orderby=None,
            table: bool=True,
            fig=None,
            **kwargs):
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
    :param kwargs:
    :return: matplotlib figure
    """

    # if no dataframe is supplied, create one
    if data is None:
        x, y, legend, data = components.create_df(x, y, legend)

    local_data = data.copy()
    local_data = local_data.reset_index()
    local_data[x] = local_data[x].astype('str')
    local_data[y] = local_data[y].astype('float').dropna()

    if orderby:
        temp = local_data.sort(x)
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

    min_, max_ = np.min(local_data[y]), np.max(local_data[y])

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(cumprob, legend, table, fig=fig)

    if orderby:
        local_data.boxplot(column=y, by=x, ax=axm, showfliers=False, positions=order, fontsize=8, **kwargs)
    else:
        local_data.boxplot(column=y, by=x, ax=axm, showfliers=False, fontsize=8, **kwargs)

    # We need to identify all of the unique entries in the groupby column
    unique_groups = set(local_data[x])
    nonan_grps = []
    for group in unique_groups:
        if 'nan' not in group:
            nonan_grps.append(group)

    if legend:
        # colormap is supposed to be the goto function to get all colormaps
        # should return a colorgrid that maps each point to a set of colors
        cgrid = common.colors.colormap(local_data[legend], kind='discrete', cmap=cmap)

        legend_color = {}
        for i, key in local_data[legend].iteritems():
            legend_color[key] = cgrid[i]
        axl = components.legend(sorted(list(legend_color.items())), axl)
        axl.set_title(legend, loc='left')

    # add all the point level data
    groups = sorted(nonan_grps)
    for j, val in enumerate(groups):
        ys = local_data[y][local_data[x] == val]
        if orderby:
            pos = x_to_loc[val]
            xs = np.random.normal(pos, 0.05, size=len(ys))
        else:
            # create the jitters for the points
            xs = np.random.normal(j + 1, 0.05, size=len(ys))

        if points:
            if legend:
                cs = cgrid[local_data[x] == val]
                axm.scatter(xs, ys.values, color=cs, marker=marker, alpha=alpha, linewidths=1, **kwargs)
            else:
                axm.scatter(xs, ys.values, marker=marker, alpha=alpha, linewidths=1, **kwargs)

        if cumprob:
            if legend:
                cs = cgrid[local_data[x] == val]
                axc = components.cumprob(ys, axc, color=cs, alpha=alpha, swapxy=True)
            else:
                axc = components.cumprob(ys, axc, alpha=alpha, swapxy=True)

    # various formating
    axm.set_ylim(min_, max_)
    axm.set_yscale(yscale)
    axm.set_ylabel(y)
    for label in axm.get_xticklabels():
        label.set_rotation(90)

    if cumprob:
        axc.set_ylim(min_, max_)
        axc.set_yscale(yscale)
        axc.set_yticklabels([], visible=False)

        for label in axc.get_xticklabels():
            label.set_rotation(90)

    if table:
        axt = components.datatable(y, data, axt, by=x)

    axm.set_title('')
    fig.suptitle('')

    return canvas.figure

if __name__ == "__main__":
    nsamples = 250
    xc = np.linspace(0, 100, nsamples)
    xc2 = xc**2
    xd = np.random.choice([1, 3, 5, 7], nsamples)
    xe = np.random.choice([10, 30, 50], nsamples)
    xf = np.random.choice([.1, .4], nsamples)
    xz = np.random.choice([np.nan], nsamples)
    xg = np.random.normal(size=nsamples)*15

    X = np.column_stack((xc, xc2, xd, xe))
    beta = np.array([1, .01, 17, .001])

    e = np.random.normal(size=nsamples)*10
    ytrue = np.dot(X, beta)
    y = ytrue + e

    data = {}
    data['xc'] = xc
    data['xc2'] = xc2
    data['xd'] = xd
    data['xe'] = xe
    data['xf'] = xf
    data['xg'] = xg
    data['y'] = y
    data['ytrue'] = ytrue

    df = pd.DataFrame.from_dict(data)

    fig = boxplot(x='xd', y='y', data=df, legend='xd', cumprob=True)
