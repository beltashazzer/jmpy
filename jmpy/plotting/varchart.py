import pandas as pd

from jmpy.plotting import boxplot
from jmpy.plotting import components


def varchart(x: list,
             y: str,
             data: pd.DataFrame,
             legend=None,
             cumprob: bool=False,
             fig=None,
             **kwargs):
    """
    varchart function
    :param x: list of strings
    :param y: str
    :param data: pd.Dataframe, source of data
    :param legend: str, color code by this column
    :param cumprob, turn on or off the cumprob plots
    :param table: turn on or off the datatable
    **kwargs: other parameters to pass into the jumpy.boxplot function
    """

    local_data = data.copy()
    local_data = local_data.reset_index()

    strx = str(x)
    strl = None

    # ensure blox plot x axis is a string and y data is all float
    for var in x:
        local_data[var] = local_data[var].astype('str')

    local_data[y] = local_data[y].astype('float').dropna()

    # join all x's into a single column
    for i, part in enumerate(x):
        if i == 0:
            local_data[strx] = local_data[part].map(str)
        else:
            local_data[strx] += local_data[part].map(str)

        if i < len(x) - 1:
            local_data[strx] += ', '

    # create a new legend column if legend is an array
    if legend and isinstance(legend, str):
        strl = legend
    elif str(legend) == strx:
        strl = strx
    elif legend:
        # make a column that has the concatenated legend label
        strl = str(legend)
        for i, part in enumerate(legend):
            if i == 0:
                local_data[strl] = local_data[part].map(str)
            else:
                local_data[strl] += local_data[part].map(str)

            if i < len(legend) - 1:
                local_data[strl] += ', '

    if fig:
        fig = boxplot(x=strx, y=y, data=local_data, orderby=strx,
                      legend=strl, cumprob=cumprob, fig=fig, **kwargs)
    else:
        fig = boxplot(x=strx, y=y, data=local_data, orderby=strx,
                      legend=strl, cumprob=cumprob, **kwargs)

    axm, axc, axl, axt = components.get_axes(fig, clear=False)

    yvals = axm.get_ylim()
    colors = ['w', 'b', 'r', 'g', 'c']

    # this array holds all the multple numbers for when to draw a line.
    # % operator is called on each of these
    nummods = [len(set(local_data[x[k]])) for k in range(len(x)) if k >= 1]
    for j, k in enumerate(reversed(range(len(nummods)))):
        if j == 0:
            nummods[k] = nummods[k]
        else:
            nummods[k] *= nummods[k-1]

    # generate every possible permutation of the incoming arrays
    for i, combo in enumerate(sorted(set(local_data[strx])), start=1):
        # draw in vertical lines
        for j, mod in enumerate(reversed(nummods)):
            if not i % mod:
                axm.vlines(i-.5, *yvals, color='{}'.format(colors[j+1]),
                           alpha=.5)

    axm.set_ylim(*yvals)
    fig.suptitle('')

    return fig
