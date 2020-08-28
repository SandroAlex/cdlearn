"""
===============================================================================
Statistics

Basic linear statistics for data sets.
===============================================================================
"""

# Load packages.
import time
import progressbar
import bottleneck

import numpy as np
import xarray as xr
import scipy as sp
import pymannkendall as mk

from importlib import reload
from statsmodels.tsa.stattools import adfuller

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# Functions.
###############################################################################
def time_correlation(
        data_array1, 
        data_array2, 
        lag=0,
        method="pearson"
    ):
    """
    Calculate correlation coefficient (Pearson or Spearman) between two input 
    data sets along time dimension for each location.
    
    Parameters
    ----------
    data_array1 : xarray DataArray object
        First input data variable.
    data_array2 : xarray DataArray object
        Second input data variable.
    lag : int, optional, default is 0
        Lag between the two variables. Please note that the first variable 
        is shifted ahead the second one in time.
    
    Returns
    -------
    xarray DataArray object
        Pearson correlation coefficient along time dimension for each pixel.
        It also returns the p-values.
    """
    
    # Time, latitude, and longitude.
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array1) 

    # Standard form.
    data_array1 = cdlearn.utils.organize_data(data_array1)
    data_array2 = cdlearn.utils.organize_data(data_array2)

    # Guarantee data alignment.
    data_array1, data_array2 = xr.align(data_array1, data_array2, join="inner")
    
    # Lagging if necessary.
    n = data_array1.sizes[dim0]
    da1_lag = data_array1.isel({dim0:slice(lag, n)}) 
    da2_lag = data_array2.isel({dim0:slice(0, n - lag)})
    
    # In the same temporal grid if occurred lagging.
    da2_lag = da2_lag.assign_coords({dim0: da1_lag.time})
    data_array1, data_array2 = da1_lag.copy(deep=True), da2_lag.copy(deep=True)

    # Extract data as numpy arrays. Each column is a time series.    
    X = data_array1.values.reshape((data_array1.shape[0], -1))    
    Y = data_array2.values.reshape((data_array2.shape[0], -1))

    # Initialize results array.
    res_values = np.zeros((2, X.shape[1]))

    # Select method.
    if method == "pearson":
        function = sp.stats.pearsonr

    elif method == "spearman":
        function = sp.stats.spearmanr    

    # Loop over locations.
    for loc in range(X.shape[1]):

        # Fill results array.
        res_values[:, loc] = function(X[:, loc], Y[:, loc])
    
    # Reshape as (stats, latitude, longitude).
    res_values = res_values.reshape(
        (2, data_array1.shape[1], data_array1.shape[2])
    )

    # Put results as an xarray DataArray object.
    results = xr.DataArray(
        data=res_values,
        dims=("stats", dim1, dim2),
        coords={"stats": ["rho", "p_value"],
                dim1: getattr(data_array1, dim1),
                dim2: getattr(data_array1, dim2)}
    )

    return results

###############################################################################
def linear_regression(
        data_array,
        verbose=False
    ):
    """
    Applies a linear least-squares regression for each location using 
    `scipy.stats.linregress` function.
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data.

    Returns
    -------
    xarray DataArray object
        It contains, for each location grid point, the following statistical 
        results: slope, intercept, r_value, p_value, std_err.
    """

    # Prepare data for the analysis.
    data_array = cdlearn.utils.organize_data(data_array)
        
    # Time, latitude, and longitude (strings).
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array) 
    
    # Extract data as numpy arrays.
    Y = data_array.values               # Dependent variable in regression. 
    X = np.arange(Y.shape[0])           # Independent variable in regression.
    Ycol = Y.reshape((Y.shape[0], -1))  # Colapse latitude and longitude.

    # Statistical results.
    r = np.nan * np.zeros((5, Ycol.shape[1]))

    if verbose:
        print(">>> Loop over grid points ...")
        time.sleep(1)
        bar = progressbar.ProgressBar(max_value=Ycol.shape[1])

    # Loop over locations.
    for i in range(Ycol.shape[1]):
        
        # Aggregate results.
        # slope, intercept, r_value, p_value, std_err.
        r[:, i] = sp.stats.linregress(X, Ycol[:, i])

        if verbose:
            bar.update(i) 

    # New shape: (results, latitude, longitude).    
    r = r.reshape((5, Y.shape[1], Y.shape[2]))
    
    # Put results as an xarray DataArray object.
    results = xr.DataArray(
        data=r,
        dims=("stats", dim1, dim2),
        coords={"stats": ["slope", "intercept", "r_value", 
                          "p_value", "std_err"],
                dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )

    # Maintain land mask coordinate into results.
    if "land_mask" in data_array.coords:
        results.coords["land_mask"] = data_array.land_mask
    
    return results

###############################################################################
def linear_trends(
        data_array
    ):
    """
    Calculate linear trends and its coefficients for data at each location.

    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data.

    Returns
    -------
    tuple of xarray DataArray object
        Linear trends and Angular and linear coefficients. 
    """
    
    # Time, latitude, and longitude (strings).
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array) 
    ndims = [dim0, dim1, dim2]

    # Standard format.
    data_array = cdlearn.utils.organize_data(data_array)     

    # New dimensions. Exclude "time" and insert "coefficients".
    dims = ndims.copy()
    dims.remove(dim0)
    dims = ["coefficients"] + dims
    
    # Extract data as numpy arrays.
    Y = data_array.values              # Dependent variable.
    X = np.arange(Y.shape[0])          # Independent variable. 
    Ycol = Y.reshape((Y.shape[0], -1)) # Colapse latitude/longitude. 
    
    # Linear fits.   
    parameters = np.polyfit(X, Ycol, deg=1)
            
    # Coefficients as xarray DataArray object.
    coefficients = xr.DataArray(
        data=parameters.reshape((2, *Y.shape[1:])),
        dims=dims,
        coords={"coefficients": ["angular", "linear"], 
                dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )
    
    # Loop over all fits (locations).
    trends = np.zeros_like(Ycol)
    for i in range(parameters.shape[1]):
        p = parameters[:, i]
        function = np.poly1d(p)
        trends[:, i] = function(X)
           
    # The same shape as the original data.
    trends = trends.reshape(*Y.shape)
        
    # Results as xarray DataArray object.
    trends = xr.DataArray(data=trends, 
                          dims=ndims, 
                          coords=data_array.coords)
    
    return trends, coefficients
                           
###############################################################################
def linear_detrend(
        data_array
    ):
    """
    This function makes a linear detrend of data for each location.
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data.

    Returns
    -------
    xarray DataArray object
        Linearly detrended data.
    """
    
    # Subtract linear fit from observed data.
    trends, _ = linear_trends(data_array)
    detrended = data_array - trends
    
    return detrended      
        
###############################################################################
def climatology(
        data_array
    ):
    """
    Calculate monthly climatology from linearly detrended data.
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data.

    Returns
    -------
    xarray DataArray object
        Monthly means from detrended data.
    """
    
    dim0, _, _ = cdlearn.utils.normalize_names(data_array) 

    # Input for grouping.
    detrended = linear_detrend(data_array)
    
    # Group data. This object has only twelve time values.
    grouped = detrended.groupby(dim0 + ".month").mean(dim0)
    
    # Initialize results.
    seasonal = xr.DataArray(np.zeros_like(data_array),
                            dims=data_array.dims,
                            coords=data_array.coords)
    
    # Filling results.
    for month in range(1, 13):
        seasonal.loc[seasonal.time.dt.month == month] = \
        grouped.loc[grouped.month == month]
        
    return seasonal

###############################################################################
def anomalies(
        data_array
    ):
    """
    Calculate anomalies as proposed by Papagiannopoulou et al (2017).
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data.

    Returns
    -------
    xarray DataArray object
        Monthly means of detrended data subtracted from detrended observed 
        data.
    
    References
    ----------
    [1] Papagiannopoulou, C.et al. 2017 : A non-linear Granger-causality 
    framework to investigate climate—vegetation dynamics.Geosci. Model 
    Dev.10:1945–1960.     
    """
    
    detrended = linear_detrend(data_array)
    seasonal = climatology(data_array)
    residuals = detrended - seasonal
    
    return residuals

###############################################################################
def daily_detrended_climatologies(
        data_array
    ):
    """
    Calculate daily anomalies from daily data. First, this function detrend 
    observed daily data. Then, typical values for each day of year are 
    calculated, resulting in 366 values. After that, these typical values are 
    subtracted of the respective detrended daily data, resulting in daily 
    climatology for detrended data.
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed daily data, not detrended.
    
    Returns
    -------
    xarray DataArray object
        Daily climatologies of daily detrended data.
    """
    
    dim0, _, _ = cdlearn.utils.normalize_names(data_array) 

    # Detrend daily data.
    detrended = linear_detrend(data_array)
    
    # Subtract each daily data value by the respective daily mean.
    grouped = detrended.groupby(dim0 + ".dayofyear")
    seasonal = grouped.mean(dim0)
    residuals = grouped - seasonal
    
    return residuals

###############################################################################
def unit_root_test(
        data_array
    ):
    """
    Check if the time series data is has unit root for each grid location.
    
    Parameters
    ---------_
    data_array : xarray DataArray object
        Data to be tested.

    Returns
    ------_
    xarray DataArray object
        Statistics containing ADF test statistic, p value, number of lags 
        used and number of observations used for the ADF regression and 
        calculation of the critical values.
    """    

    # Names and order for dimensions. Time, latitude and longitude.
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array)
                
    # Standard format.
    data_array = cdlearn.utils.organize_data(data_array)

    # Original shape.
    initial_shape = data_array.shape

    # Each pixel time series is a column in this numpy 2d array.
    X = data_array.values.reshape((initial_shape[0], -1))

    # Results initialization. 
    res_values = np.zeros(shape=(4, X.shape[1]))

    # Test for each pixel.
    for loc in range(X.shape[1]):
        
        # It does not work if data has nans.
        time_series = X[:, loc]
        mask_nan = ~np.isnan(time_series)
        time_series = time_series[mask_nan]
    
        # We have enough data.
        if len(time_series) >= int(2/3*initial_shape[0]):
        
            # Statistics. 
            adf, pvalue, usedlag, nobs, _, _ = adfuller(
                x=time_series, store=False, regresults=False
            )
            
        # We DON'T have data.    
        else:
            adf, pvalue, usedlag, nobs = np.nan, np.nan, np.nan, np.nan 

        # Fill results.
        res_values[:, loc] = adf, pvalue, usedlag, nobs

    # New shape: (results, latitude, longitude).    
    res_values = res_values.reshape((4, initial_shape[1], initial_shape[2]))
    
    # Put results as an Xarray DataArray object.
    results = xr.DataArray(
        data=res_values,
        dims=("stats", dim1, dim2),
        coords={"stats": ["adf_test", "p_value", "used_lag", "n_obs"],
                dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )

    # Add sea land mask if it is present in data_array input.
    if "land_mask" in data_array.coords:
        results = results.assign_coords({"land_mask": data_array.land_mask})

    return results

###############################################################################
def mann_kendall_tests(
        data_array,
        method="original",
        period=12,
        alpha=0.05,
        verbose=False       
    ):
    """
    
    Parameters
    ----------
    data_array : xarray DataArray object
    
    Returns
    -------
    xarray DataArray object
    
    References
    ----------
    [1] Hipel, K. W., & McLeod, A. I. (1994). Time series modelling of water 
    resources and environmental systems (Vol. 45). Elsevier.
    [2] Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of trend 
    analysis for monthly water quality data. Water resources research, 18(1),
    107â€“121. doi:10.1029/WR018i001p00107
    """

    # Put data in a commom format for calculations. Just guarantee dimensions
    # ordering (time, latitude, longitude) and ascending order for 
    # coordinates of these dimensions.
    data_array = cdlearn.utils.organize_data(data_array)
        
    # Extract data as numpy arrays.
    Y = data_array.values               # Dependent variable in regression. 
    Ycol = Y.reshape((Y.shape[0], -1))  # Colapse latitude and longitude.

    # Trend results (strings).
    r1 = np.chararray(shape=(1, Ycol.shape[1]), itemsize=12, unicode=True)
    
    # Presence of trends results (booleans).
    r2 = np.zeros((1, Ycol.shape[1]), dtype=np.bool)
    
    # Statistical results (floats).
    r3 = np.nan * np.zeros((6, Ycol.shape[1]))

    # Select method.
    if method == "original":
        
        # Original Mann-Kendall test is a nonparametric test, which does not 
        # consider serial correlation or seasonal effects.
        function = mk.original_test
        params = {"alpha": alpha}
        
    if method == "seasonal_test":
        
        # For seasonal time series data, Hirsch, R.M., Slack, J.R. and Smith,
        # R.A. (1982) proposed this test to calculate the seasonal trend.
        function = mk.seasonal_test
        params = {"period": period, "alpha": alpha}

    if method == "correlated_seasonal_test":
        
        # This method proposed by Hipel (1994) used, when time series 
        # significantly correlated with the preceding one or more 
        # months/seasons.
        function = mk.correlated_seasonal_test
        params = {"period": period, "alpha": alpha}

    if verbose:
        print(">>> Loop over grid points (%s method) ..." %(method))
        time.sleep(1)
        bar = progressbar.ProgressBar(max_value=Ycol.shape[1])    
        
    # Loop over locations.
    for i in range(Ycol.shape[1]):
        
        # All results:
        #  trend:   tells the trend (increasing, decreasing or no trend)
        #  h:       True (if trend is present) or False (if trend is absence)
        #  p_value: p-value of the significance test
        #  z:       normalized test statistics
        #  tau:     Kendall Tau
        #  s:       Mann-Kendal's score
        #  var_s:   Variance S
        #  slope:   sen's slope
        trend, h, p_value, z, tau, s, var_s, slope = \
            function(Ycol[:, i], **params)
        
        # Results.
        r1[:, i] = trend
        r2[:, i] = h
        r3[:, i] = (p_value, z, tau, s, var_s, slope)

        if verbose:
            bar.update(i) 

    # New shape: (latitude, longitude).    
    r1 = r1.reshape((Y.shape[1], Y.shape[2]))    
    r2 = r2.reshape((Y.shape[1], Y.shape[2]))
    
    # New shape: (results, latitude, longitude).    
    r3 = r3.reshape((6, Y.shape[1], Y.shape[2]))
    
    # Put trend results (strings) as an xarray DataArray object.
    trend_direction = xr.DataArray(
        data=r1,
        dims=(dim1, dim2),
        coords={dim1: getattr(data_array, dim1), 
                dim2: getattr(data_array, dim2)}
    )

    # Put presence of trends results (booleans) as an xarray DataArray object. 
    trend_presence = xr.DataArray(
        data=r2,
        dims=(dim1, dim2),
        coords={dim1: getattr(data_array, dim1), 
                dim2: getattr(data_array, dim2)}
    )    
    
    # Put statistical results (floats) as an xarray DataArray object.
    stats = xr.DataArray(
        data=r3,
        dims=("stats", dim1, dim2),
        coords={"stats": ["p_value", "z", "tau", "s", "var_s", "slope"],
                dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )
    
    # Maintain land mask coordinate into results.
    if "land_mask" in list(data_array.coords):
        trend_direction.coords["land_mask"] = data_array.land_mask
        trend_presence.coords["land_mask"] = data_array.land_mask
        stats.coords["land_mask"] = data_array.land_mask
    
    return trend_direction, trend_presence, stats        

############################################################################### 
def power_spectral_density(
        data_array,
        sample_frequency=15,
        verbose=False
    ):
    """
    To do!
    """

    # Prepare data for the analysis.
    data_array = cdlearn.utils.organize_data(data_array)

    # Time, latitude, and longitude (strings).
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array)

    # Detrend all time series.
    data_array = cdlearn.statistics.linear_detrend(data_array)

    # DatatimeIndex object
    time_index = data_array.time.to_index()
    
    # Time series size.
    N = len(time_index)
    
    # Frequencies corresponding to the values of the PSD.
    fftfreq = sp.fftpack.fftfreq(
        N, sample_frequency / 365.0 # Frequency in years^-1.
    )    
    
    # Extract data as numpy arrays.
    Y = data_array.values              # Time series. 
    Ycol = Y.reshape((Y.shape[0], -1)) # Colapse latitude and longitude.
    
    # Output results.
    results_values = np.nan * np.zeros((N, Ycol.shape[1]))
    
    if verbose:
        print(">>> Loop over grid points ...")
        time.sleep(1)
        bar = progressbar.ProgressBar(max_value=Ycol.shape[1])
    
    # Loop over grid points.
    for loc in range(Ycol.shape[1]):
        
        # Time series.
        ts = Ycol[:, loc]
        
        # FFT of the signal.
        ts_fft = sp.fftpack.fft(ts)

        # Power spectral density (PSD).
        ts_psd = np.abs(ts_fft) ** 2

        # Fill results.    
        results_values[:, loc] = ts_psd
      
        if verbose:
            bar.update(loc) 

    # New shape: (psd, latitude, longitude).    
    results_values = results_values.reshape((N, Y.shape[1], Y.shape[2]))
    
    # Put results as an Xarray DataArray object.
    results = xr.DataArray(
        data=results_values,
        dims=("freq", dim1, dim2),
        coords={"freq": fftfreq,
                dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )
    
    return results

###############################################################################
def standard_index_monthly(data_array, robust=False):
    """
    Scale data by means and standard deviations for each month.
    
    Parameters
    ----------
    data_array : xarray DataArray object
        Observed data at monthly scale.
    robust : bool, optional, default is False.
        If True, use median as average and inter quantile as deviation. Default 
        is to use mean.    

    Returns
    -------
    data_array_scaled : xarray DataArray object
        Normalized data.
    """
    
    # Prepare data for the analysis.
    data_array = cdlearn.utils.organize_data(data_array)
    
    # Time, latitude, and longitude (strings).
    dim0, _, _ = cdlearn.utils.normalize_names(data_array)
    
    # Group data. These objects have only twelve time values.
    if robust:
        grouped_avg = data_array.groupby(dim0 + ".month").median(dim0)
        q25 = data_array.groupby("time.month").quantile(0.25, dim="time")
        q75 = data_array.groupby("time.month").quantile(0.75, dim="time")
        grouped_dev = q75 - q25
        
    else:   
        grouped_avg = data_array.groupby(dim0 + ".month").mean(dim0)
        grouped_dev = data_array.groupby(dim0 + ".month").std(dim0)
    
    grouped_dev = data_array.groupby(dim0 + ".month").std(dim0)
        
    # Initialize results.
    data_array_scaled = xr.DataArray(
        data=np.nan * np.zeros_like(data_array),
        dims=data_array.dims,
        coords=data_array.coords
    )
    
    # Fill results.
    for month in range(1, 13):
        mask_month = data_array.time.dt.month == month
        DATA = data_array.sel(time=data_array.time[mask_month])
        AVG = grouped_avg.sel(month=month)
        DEV = grouped_dev.sel(month=month)
        data_array_scaled.loc[data_array_scaled.time.dt.month == month] = \
            (DATA - AVG) / DEV 
        
    return data_array_scaled

###############################################################################
def covariance_gufunc(x, y):
    """
    http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization
    """

    f1 = x - x.mean(axis=-1, keepdims=True) 
    f2 = y - y.mean(axis=-1, keepdims=True)
    cov = (f1 * f2).mean(axis=-1)
    
    return cov

###############################################################################
def pearson_correlation_gufunc(x, y):
    """
    http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization
    """
    
    n = covariance_gufunc(x, y)
    d = x.std(axis=-1) * y.std(axis=-1)
    cor = n / d
    
    return  cor

###############################################################################
def spearman_correlation_gufunc(x, y):
    """
    http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization
    """

    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    spe_cor = pearson_correlation_gufunc(x_ranks, y_ranks)
    
    return spe_cor 

###############################################################################
def xarray_spearman_correlation(x, y, dim):
    """
    http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization
    """
    
    res = xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask="parallelized",
        output_dtypes=[float]
    )
    
    return res