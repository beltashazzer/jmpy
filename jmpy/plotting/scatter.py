import pandas as pd
import numpy as np

import scipy.interpolate as spi
import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy import common
from jmpy.plotting import components


def scatter(x,
            y,
            data: pd.DataFrame=None,
            legend=None,
            marker: str='o',
            alpha: float=.5,
            xscale: str='linear',
            yscale: str='linear',
            cmap: str='default',
            figsize: tuple=(12, 6),
            fit: str=None,
            fitparams: dict=None,
            table: bool=True,
            fig=None,
            **kwargs):
    """
    Scatter plots with regression lines
    :param x:  str or ndarray
    :param y: str or ndarray
    :param data: pandas.Dataframe
    :param legend: str or ndarray, color/fit by this column
    :param marker: matplotlib marker style
    :param alpha: float, matplotlib alpha
    :param xscale: default == linear, any of matplotlib scale types
    :param yscale: default == linear, any of matplotlib scale types
    :param cmap: any of matplotlib cmaps
    :param figsize: default == (9,6);
    :param fit: [linear, quadratic, smooth, interpolate]
    :param fitparams: params to pass to fitting function
    :param table:  show the regression table
    :param kwargs:
    :return: fig, (axes)
    """

    # if no dataframe is supplied, create one
    if data is None:
        x, y, legend, data = components.create_df(x, y, legend)

    if not fitparams:
        fitparams = {}

    local_data = data.copy()
    local_data = local_data[[i for i in (x, y, legend) if i]]
    local_data = local_data.dropna()
    local_data.sort(x)
    local_data = local_data.reset_index()

    makefitaxis = False
    if fit == 'linear' or fit == 'quadratic':
        makefitaxis = True

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.create_axes(False, legend, table and makefitaxis, fig=fig)

    if legend:
        # colormap is supposed to be the goto function to get all colormaps
        # should return a colorgrid that maps each point to a set of colors
        cgrid = common.colors.colormap(local_data[legend],
                                       kind='discrete', cmap=cmap)

        legend_color = {}
        for i, key in local_data[legend].iteritems():
            legend_color[key] = cgrid[i]

        components.legend(sorted(list(legend_color.items())), axl)
        axl.set_title(legend, loc='left')

        text = ''
        for l in sorted(set(local_data[legend])):
            t = local_data[local_data[legend] == l]
            axm.scatter(x=t[x], y=t[y], c=legend_color[l],
                        marker=marker, alpha=alpha, **kwargs)

            if fit:
                xs, ys, fn = _get_fit(x, y, t, fit, fitparams)
                axm.plot(xs, ys, c=legend_color[l])

                if makefitaxis and table:
                    text += '${}:  {}$\n'.format(str(l).strip(), fn)

        if makefitaxis and table:
            components.regressiontable(text, axt, fig)
            axt.axis('off')

    else:
        axm.scatter(x=local_data[x], y=local_data[y],
                    marker=marker, alpha=alpha, **kwargs)
        if fit:
            xs, ys, fn = _get_fit(x, y, local_data, fit, fitparams)
            axm.plot(xs, ys)

            if makefitaxis and table:
                components.regressiontable('{}'.format(fn), axt, fig)

    axm.set_xlim(np.min(local_data[x]), np.max(local_data[x]))
    axm.set_ylim(np.min(local_data[y]), np.max(local_data[y]))
    axm.set_yscale(yscale)
    axm.set_xscale(xscale)
    axm.set_xlabel(x)
    axm.set_ylabel(y)

    return canvas.figure


def _get_fit(x, y, df, fit, fitparams):
    """
    Internal method to return fitted data given an x and y and datatable

    :param x: x param
    :param y: y param
    :param df: data table
    :param fit: type of fit
    :return: subsample of data and predicted line
    """

    xhat = np.linspace(df[x].min(), df[x].max(), num=100)

    if fit == 'linear':
        xs, ys = _medianify(df, x, y)
        mb = np.polyfit(xs, ys, 1, **fitparams)
        fit_fn = np.poly1d(mb)
        # TODO: make this handle precision correctly
        eq = 'f(x) = {:.4f}x + {:.4f}'.format(
            fit_fn.coeffs[0], fit_fn.coeffs[1])

        return xhat, fit_fn(xhat), eq

    elif fit == 'quadratic':
        xs, ys = _medianify(df, x, y)
        mb = np.polyfit(xs, ys, 2, **fitparams)
        fit_fn = np.poly1d(mb)
        # TODO: make this handle precision correctly...
        eq = 'f(x) = {:.4f}x^2 + {:.4f}x + {:.4f}'.format(
            fit_fn.coeffs[0], fit_fn.coeffs[1], fit_fn.coeffs[2])

        return xhat, fit_fn(xhat), eq

    elif fit == 'smooth':
        xs, ys = _medianify(df, x, y)
        xhat = np.linspace(xs.min(), xs.max(), num=100)
        spl = spi.UnivariateSpline(xs, ys, **fitparams)
        return xhat, spl(xhat), None

    elif fit == 'interpolate':
        xs, ys = _medianify(df, x, y)
        f = spi.interp1d(xs, ys, **fitparams)
        return xhat, f(xhat), None


def _medianify(df, x, y):
        t = df[[x, y]]
        # univariate spline chokes if there are multiple values per "x" so
        # we will take the median of all the doubled up x values.
        summ = t.groupby(x).agg(np.median)
        summ = summ.unstack()
        summ = summ.reset_index()
        summ = summ.sort(x)

        return summ[x], summ[0]
