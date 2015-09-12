from jmpy import common
import pandas as pd
import matplotlib as mpl
import matplotlib.gridspec as gs

def cumprob(x, ax, color=None, marker='.', alpha=1, swapxy=False):
    """
    Create a cumprob plot on a provided axes
    """

    sigma, pct, vals = common.stats.pct_sigma(x)
    if swapxy:
        vals, pct = pct, vals
    ax.scatter(vals, pct, color=color, marker=marker, alpha=alpha, )
    ax.plot(vals, pct)
    if swapxy:
        ax.set_xlim(0, 1)
    else:
        ax.set_ylim(0, 1)
    return ax


def regressiontable(text, ax, fig):
    """
    Creates the table for the scatter plot.
    Pass in text and you get a axis back.
    """
    ax.text(0, 1, '{}'.format(text),
            horizontalalignment='left',
            verticalalignment='top',
            family='Courier New',
            fontsize=12,
            transform=ax.transAxes
            )

    ax.set_axis_bgcolor('white')
    ax.grid(b=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim([0, 1])
    ax.tick_params(top="off")
    ax.tick_params(bottom="off")
    ax.tick_params(right="off")
    ax.tick_params(left="off")

    return ax


def datatable(x, data, ax, by=None):
    """
    Create a datatable on a provided axes
    """

    headers = ['Mean', 'Stdev', 'Min', '10%', '50%', '90%', 'Max', 'N']

    stats = common.stats.array_stats(x, data=data, by=by)

    groups = []
    values = []
    for group, tup in stats.items():
        groups.append(group)
        array = [round(tup[val], 3) for val in (0, 1, 2, 3, 4, 5, 6, 7)]
        values.append(array)

    tab = ax.table(cellText=values,
                   rowLabels=groups,
                   colLabels=headers,
                   loc='upper left',
                   transform = ax.transAxes)

    tab.set_fontsize(8)
    ax.set_axis_bgcolor('white')
    ax.grid(b=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim([0, 1])

    return ax


def legend(labelcolor, axl):
    """
    creates a legend axes given a list of name, color tuple and an axes name
    """
    for label, color in labelcolor:
        axl.scatter(1, 1, c=color, label=label)
        axl.set_ylim(0, .1)
        axl.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0,
                   scatterpoints=1, frameon=False)
        axl.axis('off')
    return axl


def create_axes(cumprob, legend, dt, fig):
    """
    Private method to create all the axes
    """

    rows = 1
    cols = 1
    main_width = 15
    cumprob_w = 5
    legend_w = 1

    width_array = [main_width]
    height_array = [1]

    if cumprob and legend:
        cols = 3
        width_array = [main_width, cumprob_w, legend_w]
    elif cumprob:
        cols = 2
        width_array = [main_width, cumprob_w]
    elif legend:
        cols = 2
        width_array = [main_width, legend_w]

    dt_start = len(width_array)
    dt_end = dt_start + len(width_array)

    if dt:
        rows = 2
        width_array = width_array + width_array
        height_array = [2, 1]

    grid = gs.GridSpec(rows, cols,
                       width_ratios=width_array,
                       height_ratios=height_array)

    axm = fig.add_subplot(grid[0], label='axm')  # x, y, width, height
    axc, axl, axt = None, None, None

    if cumprob:
        axc = fig.add_subplot(grid[1], label='axc')

    if legend and cumprob:
        axl = fig.add_subplot(grid[2], frameon=False, label='axl', )
    elif legend:
        axl = fig.add_subplot(grid[1], frameon=False, label='axl')

    if dt:
        axt = fig.add_subplot(grid[dt_start:dt_end], frameon=False,
                              label='axt')

    for ax in axm, axc, axl, axt:
        if not ax:
            continue

        ax.tick_params(top="off")
        ax.tick_params(bottom="off")
        ax.tick_params(right="off")
        ax.tick_params(left="off")

    fig.set_tight_layout({'w_pad': -2})
    return (axm, axc, axl, axt)


def get_axes(fig, clear=True):
    """
    Private method to get all the axes from a figure,
    but put in the correct order
    """

    axes = fig.axes
    axm, axc, axl, axt = (None,) * 4

    for ax in axes:
        if ax.get_label() == 'axm':
            axm = ax
        elif ax.get_label() == 'axc':
            axc = ax
        elif ax.get_label() == 'axl':
            axl = ax
        elif ax.get_label() == 'axt':
            axt = ax

        if clear:
            ax.cla()

    return axm, axc, axl, axt


def create_df(x, y, legend=None, z=None,):
    """
    create a generic dataframe from supplie numpy arrays
    """

    mydict = {}
    mydict['x'] = x
    mydict['y'] = y

    if z is not None:
        mydict['z'] = z

    if legend is not None:
        mydict['legend'] = legend
        legend = 'legend'

    data = pd.DataFrame.from_dict(mydict)
    x = 'x'
    y = 'y'

    if z is not None:
        z = 'z'
        return x, y, z, data

    return x, y, legend, data
