import itertools

import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb
import matplotlib.gridspec as gs

from jmpy.plotting import components
from jmpy import common


def grid(rows=None, cols=None, data=None, chart=None, args=None, figsize=(8, 8), legend=None,
         cmap='default', colorbar=False):
    """ Create a grid from pandas data

    :param grid:  dictionary of x and y columns
    :param data:  pandas dataframe or none
    :param funct:  jumpy plotting fuction, specified as a lambda
                   with data source as a variable
    :param args:  argument dictionary to pass to the chart
    :param legend: color by this column
    :param figsize: tuple to set figsize
    :param cmap: matplotlib colormap to use
    :return:
    """

    df = data.copy()
    rows_array, cols_array = [], []
    try:
        cols_array = sorted(set(df[cols]))
    except (KeyError, NameError, ValueError):
        pass

    try:
        rows_array = sorted(set(df[rows]))
    except (KeyError, NameError, ValueError):
        pass

    numcols = len(cols_array) if cols_array else 1
    numrows = len(rows_array) if rows_array else 1

    numcols += 1  # add a row and column for headings
    numrows += 1

    if legend:
        numcols += 1

    fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
    canvas = mbb.FigureCanvasAgg(fig)

    if len(cols_array):
        wrs = [1] + [5 for i in cols_array]
    else:
        wrs = [1, 5]

    if legend:
        wrs += [1]

    if rows_array:
        hrs = [1] + [5 for i in rows_array]
    else:
        hrs = [1, 5]

    grid = gs.GridSpec(numrows, numcols, width_ratios=wrs,
                       height_ratios=hrs)

    if len(cols_array) > 0:
        x = 1 if len(rows_array) else 1
        for i, val in enumerate(cols_array, start=x):
            ax = fig.add_subplot(grid[0, i])
            ax.text(.5, .3, val)
            ax.axis('off')
            p = mpl.patches.Rectangle((0, 0), 1, 1,
                                      fill=True, transform=ax.transAxes, clip_on=True,
                                      fc='#C8C8C8')
            ax.add_patch(p)

    if len(rows_array) > 0:
        y = 1 if len(cols_array) else 1
        for i, val in enumerate(rows_array, start=y):
            ax = fig.add_subplot(grid[i, 0])
            ax.text(.5, .5, val, rotation=90)
            ax.axis('off')
            p = mpl.patches.Rectangle((0, 0), 1, 1,
                                      fill=True, transform=ax.transAxes, clip_on=True,
                                      fc='#C8C8C8')
            ax.add_patch(p)

    # if rows and columns are provided, we need all combinations
    # itertools product will return nothing if one of the cols/rows is None
    # so then we will default to the longest of the cols/rows
    charts = list(itertools.product(cols_array, rows_array))
    if not list(charts):
        try:
            charts = list(itertools.zip_longest(cols_array, rows_array))
        except AttributeError:  #py2
            charts = list(itertools.izip_longest(cols_array, rows_array))

    if legend:
        cgrid = common.colors.colormap(df[legend], kind='discrete', cmap=cmap)

    for x, y in charts:
        # fitler the data for the exact chart we are looking at
        tdf = df[df[cols] == x] if (x and cols) else df
        tdf = tdf[tdf[rows] == y] if (y and rows) else tdf

        if tdf.size == 0:
            continue

        # filter te color grid to match the chart data
        tc = None
        if legend:
            tc = cgrid[df[cols] == x] if (x and cols) else cgrid
            tc = tc[df[rows] == y] if (y and rows) else tc
            tc = tc.reset_index(drop=True)

        ax = fig.add_subplot(grid[rows_array.index(y) + 1 if y else 1,
                                  cols_array.index(x) + 1 if x else 1])

        # call the particular chart in provided
        if legend:
            chart(data=tdf, axes=ax, cgrid=tc, legend=legend, **args)
        else:
            chart(data=tdf, axes=ax, cgrid=tc, **args)

    if legend:
        legend_color = {}
        for i, key in df[legend].iteritems():
            legend_color[key] = cgrid[i]

        axl = fig.add_subplot(grid[1, numcols-1])
        axl = components.legend(sorted(list(legend_color.items())), axl)
        axl.set_title(legend, loc='left')

    fig.suptitle('')
    return canvas.figure
