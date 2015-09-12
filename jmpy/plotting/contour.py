import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

from jmpy.plotting import components


def contour(x,
            y,
            z,
            data: pd.DataFrame=None,
            marker: str=None,
            alpha: float=.5,
            xscale: str='linear',
            yscale: str='linear',
            cmap=None,
            ncontours: int=100,
            gridsize: int=100,
            colorbar: bool=True,
            labels: bool=False,
            figsize: tuple=(12, 6),
            filled: bool=True,
            fig=None,
            **kwargs):
    """
    Create a contour plot from x, y, ans z values
    """

    # if no dataframe is supplied, create one
    if data is None:
        x, y, z, data = components.create_df(x, y, z=z)

    df = data.copy()
    df = df[[i for i in (x, y, z) if i]]
    df = df.dropna()
    df = df.reset_index()

    if fig:
        fig = fig
        canvas = mbb.FigureCanvasAgg(fig)
        axm, axc, axl, axt = components.get_axes(fig)
    else:
        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        axm, *_ = components.create_axes(False, False, False, fig=fig)

    xi = np.linspace(np.min(df[x]), np.max(df[x]), gridsize)
    yi = np.linspace(np.min(df[y]), np.max(df[y]), gridsize)
    zi = mpl.mlab.griddata(df[x], df[y], df[z], xi, yi, interp='linear')

    if filled:
        cf = axm.contourf(xi, yi, zi, ncontours, cmap=cmap, **kwargs)
    else:
        cf = axm.contour(xi, yi, zi, ncontours, cmap=cmap, **kwargs)

    if marker:
        axm.scatter(df[x], df[y], marker=marker, color='k')

    if colorbar:
        fig.colorbar(cf)

    if labels:
        axm.clabel(cf)

    axm.set_xlim(np.min(df[x]), np.max(df[x]))
    axm.set_ylim(np.min(df[y]), np.max(df[y]))
    axm.set_yscale(yscale)
    axm.set_xscale(xscale)
    axm.set_xlabel(x)
    axm.set_ylabel(y)

    return canvas.figure
