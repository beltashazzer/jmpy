import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy.plotting import components


def contour(x, y, z, data=None, marker=None, alpha=.5,
            xscale='linear',  yscale='linear', cmap=None,
            ncontours=100, gridsize=100, colorbar=True, labels=False,
            figsize=(12, 6), filled=True, fig=None, axes=None, cgrid=None,
            axislabels=True, axisticks=True, **kwargs):
    """
    Create a contour plot from x, y, ans z values
    """

    # if no dataframe is supplied, create one
    if data is None:
        (x, y, z, _), data = components.create_df(x, y, z, _)

    df = data.copy()
    df = df[[i for i in (x, y, z) if i]]
    df = df.dropna()
    df = df.reset_index()

    if fig and not axes:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes_1(fig)
    elif axes:
        axm = axes
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, _, _, _ = components.create_axes(False, False, False, fig=fig)

    xi = np.linspace(np.min(df[x]), np.max(df[x]), gridsize)
    yi = np.linspace(np.min(df[y]), np.max(df[y]), gridsize)
    try:
        zi = mpl.mlab.griddata(df[x], df[y], df[z], xi, yi, interp='linear')
    except ValueError:
        return

    if filled:
        cf = axm.contourf(xi, yi, zi, ncontours, cmap=cmap, **kwargs)
    else:
        cf = axm.contour(xi, yi, zi, ncontours, cmap=cmap, **kwargs)

    if not axisticks:
        axm.get_xaxis().set_visible(False)
        axm.get_yaxis().set_visible(False)

    if marker:
        axm.scatter(df[x], df[y], marker=marker, color='k')

    if colorbar and not axes:
        fig.colorbar(cf)

    if labels:
        axm.clabel(cf)

    axm.set_xlim(np.min(df[x]), np.max(df[x]))
    axm.set_ylim(np.min(df[y]), np.max(df[y]))
    axm.set_yscale(yscale)
    axm.set_xscale(xscale)

    if axislabels:
        axm.set_xlabel(x)
        axm.set_ylabel(y)

    if axes:
        return axm
    else:
        return canvas.figure
