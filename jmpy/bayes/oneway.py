import random

import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.gridspec as gs
import matplotlib as mpl
import matplotlib.backends.backend_agg as mbb

import jmpy.plotting as jp


class JbResult(object):
    def __init__(self, model, groups):
        self.model = model
        self.df = None
        self.groups = groups
        self.names = []
        for name in ('mean', 'delta', 'deltam', 'deltas'):
            self.names += ['{}_{}'.format(name, g) for g in groups]

    def traceplot(self, name, **kwargs):
        return pm.Matplot.plot(self.model.trace(name), **kwargs)

    def result(self, boxkwargs=None, figsize=(12, 10), maxpoints=1000):
        """

        :return:
        """

        if boxkwargs is None:
            boxkwargs = {}

        filtered = random.sample(list(self.df.index), maxpoints)
        sampled_df = self.df.ix[filtered]

        fig = mpl.figure.Figure(figsize=figsize, tight_layout=True)
        canvas = mbb.FigureCanvasAgg(fig)
        grid = gs.GridSpec(4, 3,
                           width_ratios=[4, 4, 1],
                           height_ratios=[2, 1, 2, 1])

        axm = fig.add_subplot(grid[0], label='axm')
        axmd = fig.add_subplot(grid[1], label='axmd')
        axtm = fig.add_subplot(grid[3], label='axtm', frameon=False,)
        axtmd = fig.add_subplot(grid[4], label='axtmd', frameon=False,)
        axs = fig.add_subplot(grid[6], label='axs')
        axsd = fig.add_subplot(grid[7], label='axsd')
        axts = fig.add_subplot(grid[9], label='axts', frameon=False,)
        axtsd = fig.add_subplot(grid[10], label='axtsd', frameon=False,)

        jp.boxplot('Group', 'Mean', sampled_df, axes=axm,
                   legend='Group', **boxkwargs)
        jp.boxplot('Group', 'Mean Delta', sampled_df, axes=axmd,
                   legend='Group', **boxkwargs)
        jp.boxplot('Group', 'Sigma', sampled_df, axes=axs,
                   legend='Group', **boxkwargs)
        jp.boxplot('Group', 'Sigma Delta', sampled_df, axes=axsd,
                   legend='Group', **boxkwargs)

        jp.components.datatable('Mean', self.df, axtm,
                                by='Group', probs='bayes')
        jp.components.datatable('Mean Delta', self.df, axtmd,
                                by='Group', probs='bayes')
        jp.components.datatable('Sigma', self.df, axts,
                                by='Group', probs='bayes')
        jp.components.datatable('Sigma Delta', self.df, axtsd,
                                by='Group', probs='bayes')

        for ax in axm, axmd, axtm, axtmd, axs, axsd, axts, axtsd:
            if not ax:
                continue

            ax.tick_params(top="off")
            ax.tick_params(bottom="off")
            ax.tick_params(right="off")
            ax.tick_params(left="off")

        fig.suptitle('')
        return canvas.figure


def oneway(x, y, data=None, control=None, iterations=24000, burn=4000, verbose=False):
    """ Generate a oneway anova via bayesian markov-chain monte carlo

    :param x: str or ndarray, x axis (grouping axis)
    :param y: str or ndarray, y axis (values)
    :param data: pandas dataframe or Null
    :param control: If x has greater than two groups, you need to compare vs. the control
    :param iterations:
    :param burn:
    :param verbose: bool; default=False; turns on pymc progress bar
    :return: a results object.
    """

    if data is None:
        x, y, _, data = jp.components.create_df(x, y, None)

    df = data.copy()
    df = df.reset_index()
    df[x] = df[x].astype('str')
    df[y] = df[y].astype('float').dropna()

    groups = sorted(set(df[x]))

    if len(groups) > 2 and control is None:
        raise ValueError('Need a control group')
    elif len(groups) <= 2 and control is None:
        control = groups[0]

    means = {}  # prior distributions
    sigmas = {}  #
    deltas_mean = {}  # deltas vs control
    deltas_sig = {}
    obs = {}  # observations
    nu = {}

    def delta(dist, control, exp):
        return dist[exp] - dist[control]

    for group in groups:
        means[group] = pm.Uniform('mean_{}'.format(group), np.percentile(df[y], 1), np.percentile(df[y], 99))
        nu[group] = pm.Uniform('nu_{}'.format(group), 0, 1000)
        sigmas[group] = pm.Gamma('sigma_{}'.format(group), np.median(df[y].dropna()), nu[group])
        obs[group] = pm.Normal('obs_{}'.format(group), means[group],  1/sigmas[group]**2, value=df[df[x]==group][y],
                               observed=True, trace=True)

    for group in groups:
        if group == control:
            continue
        else:
            deltas_mean[group] = pm.Deterministic(delta, doc='Delta function of mean', name='deltam_{}'.format(group),
                                                  parents={'dist': means, 'exp': group, 'control': control}, trace=True)
            deltas_sig[group] = pm.Deterministic(delta, doc='Delta function of sigma', name='deltas_{}'.format(group),
                                                 parents={'dist': sigmas, 'exp': group, 'control': control}, trace=True)

    mylist = [means[i] for i in groups] + [obs[i] for i in groups] + [deltas_mean[i] for i in groups if i != control]
    mylist += [sigmas[i] for i in groups] + [nu[i] for i in groups] + [deltas_sig[i] for i in groups if i != control]

    model = pm.Model(mylist)
    map_ = pm.MAP(model, )
    map_.fit()
    mcmc = pm.MCMC(model, )
    mcmc.sample(iterations, burn=burn, progress_bar=verbose)

    res = JbResult(mcmc, groups)

    for group in groups:

        if res.df is None:
            res.df = pd.DataFrame()

            res.df['Mean'] = mcmc.trace('mean_{}'.format(group))[:]
            res.df['Sigma'] = mcmc.trace('sigma_{}'.format(group))[:]
            res.df['Group'] = group

            if group == control:
                continue
            res.df['Mean Delta'] = mcmc.trace('deltam_{}'.format(group))[:]
            res.df['Sigma Delta'] = mcmc.trace('deltas_{}'.format(group))[:]
        else:
            tdf = pd.DataFrame()
            tdf['Mean'] = mcmc.trace('mean_{}'.format(group))[:]
            tdf['Sigma'] = mcmc.trace('sigma_{}'.format(group))[:]
            tdf['Group'] = group

            if group == control:
                res.df = res.df.append(tdf)
                continue

            tdf['Mean Delta'] = mcmc.trace('deltam_{}'.format(group))[:]
            tdf['Sigma Delta'] = mcmc.trace('deltas_{}'.format(group))[:]

            res.df = res.df.append(tdf)

    return res
