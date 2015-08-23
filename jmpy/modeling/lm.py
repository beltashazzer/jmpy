import re

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import patsy
import matplotlib.gridspec as gs
import scipy.stats as sps


def fit(model: str, data: pd.DataFrame, model_type='ols', sample_rate: float=.8, figsize=(10, 10), fontsize: int=12, style='ggplot'):
    """
    Linear regression model with visualization of fitting parameters

    :param model: patsy model specification
    :param data: padas dataframe of results
    :param model_type: ols or rlm  (ordinary least squares or robust linear model)
    :param sample_rate: float (range of 0 to 1).  partions traning and testing set
    :param figsize: figure size of output
    :param fontsize: fonsize for regression output
    :return: tuple of axes.
    """

    if style:
        plt.style.use(style)

    full = data.copy()
    mask = np.random.uniform(low=0, high=1, size=len(full))

    if sample_rate < 1:
        train = data[mask <= sample_rate]
        test = data[mask > sample_rate]
    else:
        train = full

    y, x = patsy.dmatrices(model, train)
    
    # we never want to plot the intercept, so need to make space if it is missing
    has_int = 0
    for name in x.design_info.column_names:
        if name == 'Intercept':
            has_int = 1

    if model_type.lower() == 'ols':
        model = smf.ols(model, data=train,)
    elif model_type.lower() == 'rlm':
        model = smf.rlm(model, data=train)

    setattr(model.data.orig_exog, 'design_info', x.design_info)  # some bug in patsy makes me do this...
    results = model.fit()
        
    # get predict values from confirmation data set.
    if sample_rate < 1:
        yfit = results.predict(test)

        # bug in statsmodels should be fixed in 0.7.
        if yfit.shape != test[model.endog_names].shape:
            sample_rate = 1
    
    summ = results.summary2()

    var = model.endog_names  # y variable name
    
    # track all categorical variables because they are separated in the design
    # matrix.  We need to adjust the plotting later on to lump all the categoricals into one.
    categoricals = [c for c in x.design_info.column_names if c.startswith('C(')]
    cat_plots = set([re.split('[()]', x)[1] for x in categoricals])
    
    # generate linear regression line of the acdtual vs predicted plot.
    # slope, intercept, r_value, p_value, std_err = sps.linregress(y[:, 0], results.fittedvalues)
    xs = np.linspace(np.min(y[:, 0]), np.max(y[:, 0]), 100)
    
    # determine how many columns we need on the plot
    r, c = np.shape(x)
    if len(categoricals) > 0:
        c = c - len(categoricals) + len(cat_plots) - has_int
        
    # make a min of 3 columns, even if there are less factors
    if c < 3:
        c = 3
    
    fig = plt.figure(tight_layout=True, figsize=figsize)
    grid = gs.GridSpec(3, c)

    # model axis
    axm = fig.add_subplot(grid[0, 0:c-1])
    axm.scatter(y[:, 0], results.fittedvalues, label='Train')
    axm.plot(xs, xs, 'k--')
    
    # plot the sampled data along with the fitted data
    if sample_rate < 1:
        axm.scatter(test[var], yfit, label='Test', color='red')
    
    axm.set_title('Actual vs Predicted Plot')
    axm.set_ylabel('{} Predicted'.format(var))
    axm.set_xlabel(var)
    plt.setp(axm.xaxis.get_majorticklabels(), rotation=90)

    try:
        fig.text(.1, .9, '$R^2$ = {}\nRMSE = {}'.format(round(results.rsquared, 2), round(results.mse_total**.5, 4)))
    except AttributeError:
        pass

    axm.legend(scatterpoints=1, loc=4)

    # histogram axis
    axh = fig.add_subplot(grid[0, c-1])
    axh.hist(list(results.resid))
    axh.set_title('Model Residuals')
    plt.setp(axh.xaxis.get_majorticklabels(), rotation=90)
    
    # text axis
    axt = fig.add_subplot(grid[2, :])
    axt.text(0, 1, summ.as_text(),
             horizontalalignment='left', 
             verticalalignment='top', 
             family='Courier New',
             fontsize=fontsize,
             weight='semibold'
             )
    axt.axis('off')

    # plot scatter plots in factor row
    numcats = 0
    column = 1
    ski = 0
    for i, factor in enumerate(x.T):
        if x.design_info.column_names[i] == 'Intercept':  # skip the intercept
            ski = 1
            continue
        if x.design_info.column_names[i].startswith('C('):
            numcats += 1
            continue
        
        column = i - numcats - ski
            
        # draw scatter plot
        ax = fig.add_subplot(grid[1, column])
        ax.scatter(factor, y[:, 0])
        ax.set_xlabel(x.design_info.column_names[i])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

        # draw dotted linear regression line on each factor plot
        slope, intercept, r_value, p_value, std_err = sps.linregress(factor, y[:, 0])
        xs = np.linspace(np.min(factor), np.max(factor), 100)
        ys = xs * slope + intercept
        ax.plot(xs, ys, 'r--')
        ax.set_ylabel(var)

    # plot categorical plots in factor row
    for i, factor in enumerate(cat_plots, start=column+1):
        ax = fig.add_subplot(grid[1, i])
        ax = full.boxplot(column=var, by=factor, ax=ax, showfliers=False)
        ax.set_title('')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    fig.suptitle('')

    return fig, (fig.get_axes()), results


def check_estimates(model, data, model_type='ols', sample_rate=.8, iterations=100):
    """
    Function to quantify the variation in parameter estimates of a model
    """
    full = data.copy()
    
    if model_type.lower() == 'ols':
        func = smf.ols
    elif model_type.lower() == 'rlm':
        func = smf.rlm
    
    coeffs = pd.Series()
    for i in range(iterations):
        mask = np.random.uniform(low=0, high=1, size=len(full))

        dat = data[mask <= sample_rate]
        mod = func(model, data=dat)
        results = mod.fit()

        coeffs = coeffs.append(results.params)
    
    return coeffs.hist(by=coeffs.index, figsize=(9, 6))