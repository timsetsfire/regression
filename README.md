# regression
two-step full transform method for linear models with autoregressive errors structures

Suppose that

$$y = x\beta + \epsilon$$

with

$$\epsilon_t = \rho_1 \epsilon_{t-1} + \cdots \rho_r \epsilon_{t-r} + \nu_t.$$

Here $y$ is $n \times 1$, and $x$ is $n \times k$.  

This regression will perform the two-step full transform method to estimate the regression
and standard errors (same as SAS PROC AUTOREG setting method=YW)

~~~
from regression.linear_model import GLSYW
model = GLSYW(endog = y, exog = sma.add_constant(x), ar=1)
fit = model.fit()
fit.summary()
~~~

I'm working on some functions to mimic the output from SAS PROC AUTOREG

To replicate in SAS' PROC AUTOREG

~~~
PROC AUTOREG data = dataset;
  model y = x1 - xk / method = yw nlag = r;
RUN;
~~~
