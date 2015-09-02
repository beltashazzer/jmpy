
## JMPY  
Jmpy is an analysis library created to simplify common plotting and modeling tasks.  Simplicity, a common plotting signature, and ease of use are preferred over flexibility.

The goal is to create mini-reports with each function for better visualization of data.
___
Currently, there are two modules:  plotting and modeling.  
Plotting is used to make pretty graphs, and modeling is used to anaylyze data with visual output.  

This module relies heavily on statsmodels, patsy, and pandas.


```python
%load_ext autoreload
%autoreload 2
    
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import jmpy.plotting as jp
import jmpy.modeling as jm

import warnings
warnings.filterwarnings('ignore')
```

### Plotting  
___
Create some artifical data for our analysis:


```python
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
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xc</th>
      <th>xc2</th>
      <th>xd</th>
      <th>xe</th>
      <th>xf</th>
      <th>xg</th>
      <th>y</th>
      <th>ytrue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7</td>
      <td>50</td>
      <td>0.4</td>
      <td>7.628784</td>
      <td>114.007255</td>
      <td>119.050000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.401606</td>
      <td>0.161288</td>
      <td>3</td>
      <td>10</td>
      <td>0.4</td>
      <td>16.637129</td>
      <td>52.239746</td>
      <td>51.413219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.803213</td>
      <td>0.645151</td>
      <td>5</td>
      <td>50</td>
      <td>0.1</td>
      <td>-14.003879</td>
      <td>59.695922</td>
      <td>85.859664</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.204819</td>
      <td>1.451589</td>
      <td>3</td>
      <td>50</td>
      <td>0.4</td>
      <td>-2.778978</td>
      <td>42.663512</td>
      <td>52.269335</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.606426</td>
      <td>2.580604</td>
      <td>5</td>
      <td>10</td>
      <td>0.4</td>
      <td>17.615734</td>
      <td>95.051037</td>
      <td>86.642232</td>
    </tr>
  </tbody>
</table>
</div>




```python
jp.histogram('y', df)
```




![png](README_files/README_4_0.png)



Lets start to visualize how our artifical data looks.  First, plot a histogram of the results.

If you want to look at the data color coded by a categorical variable, you need to specify "legend":


```python
jp.histogram('y', df, legend='xd', bins=50, cumprob=True)
```




![png](README_files/README_7_0.png)



You can also look at variablility and cumprob plots:


```python
jp.cumprob('y', df, legend='xd', marker='D')
```




![png](README_files/README_9_0.png)




```python
jp.varchart(['xd', 'xe'], 'y', data=df, legend=['xd'], cumprob=True, figsize=(9,6))
```




![png](README_files/README_10_0.png)



___
Lets look at the data with a boxplot to see if there is any difference between the groups defined by xd:


```python
jp.boxplot(x='xd', y='y', data=df, legend='xd', cumprob=False)
```




![png](README_files/README_12_0.png)



___
You can also create a scatter plot with a fit.


```python
jp.scatter(x='xc', y='y', data=df, legend='xd', fit='linear', marker='^')
```




![png](README_files/README_14_0.png)



We can generate the same graph using the arrays directly without creating the pandas dataframe.  You can fit the data by specifying a fit param.  Currently, linear, quadratic, smooth, and interpolate are supported.


```python
jp.scatter(xc, y=y, legend=xd, fit='linear', marker='^')
```




![png](README_files/README_16_0.png)




```python
jp.scatter('xc2', 'y', df, legend='xd', fit='quadratic', marker='o')
```




![png](README_files/README_17_0.png)




```python
# fitparams get passed into the fitting functions.  Smoothing uses the scipy Univariate spline function.
jp.scatter(x='xc2', y='y', data=df, legend='xd', fit='smooth', fitparams={'s': 1e6})
```




![png](README_files/README_18_0.png)



Contour plots can also be created as well:


```python
jp.contour('xc2', 'xc', 'y', df, cmap='YlGnBu')
```




![png](README_files/README_20_0.png)



___
### Modeling  
#### Ordinary Least Squares
Now that we have visualized some of the data, lets do some modeling.  jmpy current supports two types of linear modeling:  ordinary least squares and robust linear model, all built on the statsmodels functions.  Lets do the OLS first.  

All models are specified based on the patsy text formulas that are very similar to R.  

By default, only 80% of the data is used to fit the model, and the other 20% is plotted alongside the data to validate the model results.  This can be changed by specifying a different sample_rate parameter.


```python
model = 'y ~ xc + xc2 + C(xd) +  xg'
jm.fit(model, data=df, sample_rate=.8, model_type='ols');
```


![png](README_files/README_22_0.png)


___
#### Robust Linear Model
Now... what if we had a couple of outliers in our dataset... lets create some outliers


```python
dfo = df.copy()
p = np.random.uniform(0, 1, size=dfo.y.shape)
err = np.random.normal(size=dfo.y.shape)

df['yout'] = dfo.y + np.greater_equal(p, 0.9) * (dfo.y * err)
```

Our coefficient estimates will be skewed due to the outliers.


```python
model = 'yout ~ xc + xc2 + C(xd)'
jm.fit(model, data=dfo, sample_rate=.8, model_type='ols');
```


![png](README_files/README_26_0.png)


Employing the robust linear model, we can minimize the influence of the outliers, and get better coefficient predictions.


```python
model = 'y ~ xc + xc2 + C(xd)'
jm.fit(model, data=dfo, sample_rate=.8, model_type='rlm');
```


![png](README_files/README_28_0.png)


Our parameter estimates using the robust linear model are much closer to the truth, than using the OLS.

#### Overfitting  
Next we will test how robust the parameter estimates are by running many iterations of the model, and randomly subsetting the data, and then looking at the parameter estimate distributions.


```python
model = 'y ~ xc + xc2 + C(xd)'
jm.check_estimates(model, data=df, sample_rate=.8, model_type='ols', iterations=275);
```


![png](README_files/README_31_0.png)



```python

```
