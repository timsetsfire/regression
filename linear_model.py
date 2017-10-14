import statsmodels.api as sma
from statsmodels.regression.linear_model import OLS, RegressionModel
import numpy as np

class GLSAR(RegressionModel):
    __doc__ = """
    A regression model with an AR(p) covariance structure.
    Implements the two-step full transform method Harvey (1981) aka Yule-Walker
    method in SAS

    The Yule-Walker method alternates estimation of beta using generalized
    least squares with estimation of ar terms using the Yule-Walker equations applied
    to the sample autocorrelation function. The YW method starts by forming
    the OLS estimate of beta. Next, the ar terms are estimated from the sample
    autocorrelation function of the OLS residuals by using the Yule-Walker equations.
    Then V (V is proportional to the variance matrix of the error vector, i.e. Sigma)
    is estimated from the estimate of ar terms, and Sigma is estimated from V
    and the OLS estimate of sigma^2. The autocorrelation corrected estimates of
    the regression parameters beta are then computed by GLS, using the estimated
    Sigma matrix.

    %(params)s
    ar : int
        order of the autoregressive covariance
    %(extra_params)s
    """ % {'params' : base._model_params_doc,
           'extra_params' : base._missing_param_doc + base._extra_param_doc }
    def __init__(self,endog,exog=None,ar=1,missing=None,hasconst=None,**kwargs):
        if exog is None:
            exog = np.ones((endog.shape[0],1))
        # compute initial ols
        nobs = exog.shape[0]
        ols = OLS(endog, exog).fit()
        df_resid = exog.shape[0] - exog.shape[1] - ar
        RHO = sma.regression.yule_walker(ols.resid, order=ar, inv=True, method="mle")
        yw_coef, sig, ginv = RHO
        sig *= sig ## preliminary mse
        yw_std = np.sqrt(np.diag(sig * ginv) / df_resid)

        # http://www.stat.ufl.edu/~winner/sta6208/gls1.pdf
        P = np.linalg.cholesky(ginv).T  # contructing V
        T11 = np.sqrt(sig)*P   # constructing V
        T12 = np.zeros([ar, nobs - ar])  # constructing V
        rho_reversed = np.append(-yw_coef[::-1], 1) # constructing V
        T2 = np.zeros([nobs - ar, nobs])# constructing V
        for i in range(nobs - ar): # constructing V
            T2[i, 0 + i:ar + i + 1] = rho_reversed # constructing V
        T = np.concatenate( (np.concatenate((T11, T12), axis=1), T2), axis=0) # constructing V
        V = np.dot(T.T, T) # contructing V
        V_inv = np.linalg.pinv(V)
        sigma, cholsigmainv = sma.regression.linear_model._get_sigma(V_inv, len(endog))

        self.L = (scipy.linalg.cholesky( V, lower=False)[-ar:,-ar:])[::-1,::-1]
        self.preliminary_mse = sig

        self.yw_coef = yw_coef
        self.yw_std = yw_std
        super(GLSAR, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, sigma=sigma,
                                  cholsigmainv=cholsigmainv, **kwargs)

        self.df_resid = df_resid
        self.df_model = self.nobs - self.df_resid
        self.estimated_acorr = acf(ols.resid, unbiased = False, nlags=ar+1,
                                   qstat=False,fft=False,alpha=None,missing='none')
        self.estimated_acov = self.estimated_acorr * np.linalg.pinv(ginv)[0][0]


        self._data_attr.extend(['sigma', 'cholsigmainv', 'ar', 'L', 'yw_coef'])


    def whiten(self, X):
        """
        Whitening method from SAS proc autoreg.
        Parameters
        -----------
        X : array-like
            Data to be whitened.

        Returns
        ------------
        wX : array-like
            whitened data

        See Also
        --------
        http://support.sas.com/documentation/cdl/en/etsug/63939/HTML/default/viewer.htm#etsug_autoreg_sect028.htm
        """
        # whiten first ar terms based on incomplete information.
        x_first_q_obs = np.dot(self.L, X[0:self.yw_coef.shape[0]]) ## replace coef
        X = np.asarray(X, np.float64)
        _X = X.copy()
        #the following loops over the first axis,  works for 1d and nd
        # whitens remaining n - ar observations
        for i in range(self.yw_coef.shape[0]):
            _X[(i+1):] = _X[(i+1):] - self.yw_coef[i] * X[0:-(i+1)]
        #return _X[rho.shape[0]:]
        wX = np.concatenate( (x_first_q_obs, _X[self.yw_coef.shape[0]:]), axis=0)
        return wX

    def loglike(self, params):
        """
        Returns the value of the Gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array-like
            The parameter estimates

        Returns
        -------
        loglike : float
            The value of the log-likelihood function for a GLS Model.

        Notes
        -----
        The log-likelihood function for the normal distribution is

        .. math:: -\\frac{n}{2}\\log\\left(\\left(Y-\\hat{Y}\\right)^{\\prime}\\left(Y-\\hat{Y}\\right)\\right)-\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)-\\frac{1}{2}\\log\\left(\\left|\\Sigma\\right|\\right)

        Y and Y-hat are whitened.

        """
        #TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params))**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with likelihood constant
        if np.any(self.sigma):
        #FIXME: robust-enough check?  unneeded if _det_sigma gets defined
            if self.sigma.ndim==2:
                det = np.linalg.slogdet(self.sigma)
                llf -= .5*det[1]
            else:
                llf -= 0.5*np.sum(np.log(self.sigma))
            # with error covariance matrix
        return llf
