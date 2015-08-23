from jmpy import common
import pandas as pd
import matplotlib as mpl


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
    ax.text(0, .3, '{}'.format(text),
            horizontalalignment='left',
            verticalalignment='top',
            family='Courier New',
            fontsize=12,
            transform=fig.transFigure)

    ax.set_axis_bgcolor('white')
    ax.grid(b=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim([0,1])
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
              loc='center left')

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
        axl.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0, scatterpoints=1, frameon=False )
        axl.axis('off')
    return axl


def create_axes(cumprob, legend, dt, fig):
    """
    Private method to create all the axes
    """
    
    mpl.rcParams.update({'axes.labelsize': 'small'})
    mpl.rcParams.update({'xtick.labelsize': 'small'})
    mpl.rcParams.update({'ytick.labelsize': 'small'})
    mpl.rcParams.update({'legend.fontsize': 'small'})
    mpl.rcParams.update({'font.size': 10.0})

    xmargin = 0.1 # per side
    xmmargin = 0.05 if (legend and cumprob) else 0 # margin between two axes in x
    
    ymargin = 0.05 if dt else 0.1 # top only
    ymmargin = 0.1 if dt else 0 # margin between two axes in y direction    
    
    axc_w = .2 if cumprob else 0
    axl_w = .05 if legend else 0
    axm_w = 1 - (2*xmargin) - axc_w - axl_w - xmmargin
    axt_w = 1 - (2 * xmargin)
    
    axt_h = (1 - (2*ymargin) - ymmargin) / 2 if dt else 0
    axm_h, axc_h, axl_h = ((1 - (2*ymargin) - ymmargin) - axt_h, ) * 3

    axm = fig.add_axes([xmargin, 1-axm_h-ymargin, axm_w, axm_h], label='axm') # x, y, width, height
    axc, axl, axt = None, None, None
    
    if cumprob:
        axc = fig.add_axes([xmargin + axm_w + xmmargin, 1-axm_h-ymargin, axc_w, axc_h], label='axc')

    if legend and cumprob:
       axl = fig.add_axes([xmargin + axm_w + xmmargin + axc_w, 1-axm_h-ymargin, axl_w, axl_h], frameon=False, label='axl')
    elif legend:
        axl = fig.add_axes([xmargin + axm_w , 1-axm_h-ymargin, axl_w, axl_h], frameon=False, label='axl')

    if dt:
        axt = fig.add_axes([xmargin, 0, axt_w, axt_h], frameon=False, label='axt')

    for ax in axm, axc, axl, axt:
        if not ax:
            continue
        
        ax.tick_params(top="off")
        ax.tick_params(bottom="off")
        ax.tick_params(right="off")
        ax.tick_params(left="off")

    return (axm, axc, axl, axt)
    
def get_axes(fig, clear=True):
    """
    Private method to get all the axes from a figure, but put in the correct order
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
    
def create_df(x, y, legend=None):
    """
    create a generic dataframe from supplie numpy arrays
    """
    
    mydict = {}
    mydict['x'] = x
    mydict['y'] = y
    if legend is not None:
        mydict['legend'] = legend
        legend = 'legend'

    data = pd.DataFrame.from_dict(mydict)
    x = 'x'
    y = 'y'
    
    return x, y, legend, data
    
