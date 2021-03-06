"""
===============================================================================
Statistics

Basic linear statistics for data sets.
===============================================================================
"""

# Load packages.
import time

import numpy as np
import xarray as xr

from importlib import reload
from scipy import stats
from tqdm import tqdm

# My modules.
import cdlearn.utils

# Incorporate ongoing changes.
reload(cdlearn.utils)

# Functions.
###############################################################################
def linear_regression(
        data_set,
        var_code,
        verbose=False
    ):
    """
    Ordinary Least Squares (OLS) linear regression.

    Parameters
    ----------
    data_set : xarray Dataset object
        Input data containting `time` dimension.         
    var_code : str
        Name of the variable inside `data_set` object.       
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.

    Returns
    -------
    results : xarray Dataset object
        Results of linear ordinary least squares trends. This object contains
        variables for slope, intercept, r value, p value, and stdandard error 
        for each grid point.
    """

    # Extract xarray DataArray object.
    data_array = getattr(data_set, var_code)

    # Only land pixels.
    if "land_mask" in data_set.coords:
        data_array = data_array.where(data_array.land_mask==True, drop=True)

    # Prepare data for the analysis.
    data_array = cdlearn.utils.organize_data(data_array)
        
    # Time, latitude, and longitude (strings).
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array) 
    
    # Extract data as numpy arrays.
    Y = data_array.values               # Dependent variable in regression. 
    X = np.arange(Y.shape[0])           # Independent variable in regression.
    Ycol = Y.reshape((Y.shape[0], -1))  # Colapse latitude and longitude.

    # Element-wise mask for not a numbers.
    mask_nan = np.isnan(Y)

    # Colapse latitude and longitude.
    mask_nan = mask_nan.reshape((mask_nan.shape[0], -1))

    # This mask tells me all grid points where there are no data at all.
    mask_nan = np.all(mask_nan, axis=0)
    
    # Statistical results.
    r = np.nan * np.zeros((5, Ycol.shape[1]))

    if verbose:
        print(">>> Loop over grid points ...")
        time.sleep(1)
        bar = tqdm(total=Ycol.shape[1])

    # Loop over locations.
    for i in range(Ycol.shape[1]):

        # Good to go!
        if mask_nan[i] == False:        
        
            # Aggregate results.
            # slope, intercept, r_value, p_value, std_err.
            r[:, i] = stats.linregress(X, Ycol[:, i])

        if verbose:
            bar.update(1) 

    # Close progress bar.
    if verbose:
        bar.close()

    # New shape: (results, latitude, longitude).    
    r = r.reshape((5, Y.shape[1], Y.shape[2]))
    
    # Put results as an xarray Dataset object.
    results = xr.Dataset(
        data_vars={"slopes": ((dim1, dim2), r[0, :, :]), 
                   "intercepts": ((dim1, dim2), r[1, :, :]),
                   "r_values": ((dim1, dim2), r[2, :, :]),
                   "p_values": ((dim1, dim2), r[3, :, :]),
                   "std_errs": ((dim1, dim2), r[4, :, :])},
        coords={dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}        
    )

    # Maintain land mask coordinate into results.
    if "land_mask" in data_array.coords:
        results.coords["land_mask"] = data_array.land_mask
    
    return results

###############################################################################
def theil_slopes(
        data_set, 
        var_code,
        verbose=False
    ):
    """
    Pixel-wise trends using Theil-Sen slope estimator.

    Parameters
    ----------
    data_set : xarray Dataset object
        Input data containting `time` dimension.         
    var_code : str
        Name of the variable inside `data_set` object.       
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.

    Returns
    -------
    results : xarray Dataset object
        Results of Theil-Sen estimator. This object contains variables for 
        slope and intercept for each grid point.
    """
    
    # Extract xarray DataArray object.
    data_array = getattr(data_set, var_code)
    
    # Only land pixels.
    if "land_mask" in data_set.coords:
        data_array = data_array.where(data_array.land_mask==True, drop=True)
    
    # Prepare data for the analysis.
    data_array = cdlearn.utils.organize_data(data_array)

    # Time, latitude, and longitude (strings).
    dim0, dim1, dim2 = cdlearn.utils.normalize_names(data_array) 
    
    # Extract data as numpy arrays.
    Y = data_array.values               # Dependent variable in regression. 
    X = np.arange(Y.shape[0])           # Independent variable in regression.
    Ycol = Y.reshape((Y.shape[0], -1))  # Colapse latitude and longitude.    
    
    # Element-wise mask for not a numbers.
    mask_nan = np.isnan(Y)

    # Colapse latitude and longitude.
    mask_nan = mask_nan.reshape((mask_nan.shape[0], -1))

    # This mask tells me all grid points where there are no data at all.
    mask_nan = np.all(mask_nan, axis=0)
    
    # Statistical results.
    r = np.nan * np.zeros((2, Ycol.shape[1]))
    
    if verbose:
        print(">>> Loop over grid points ...")
        time.sleep(1)
        bar = tqdm(total=Ycol.shape[1])
    
    # Loop over locations.
    for i in range(Ycol.shape[1]):
        
        # Good to go!
        if mask_nan[i] == False:
        
            # Aggregate results.
            slope, intercept, _, _ = stats.theilslopes(Ycol[:, i], x=X)
            r[0, i] = slope
            r[1, i] = intercept
                        
        if verbose:
            bar.update(1) 

    # Close progress bar.
    if verbose:
        bar.close()

    # New shape: (results, latitude, longitude).    
    r = r.reshape((2, Y.shape[1], Y.shape[2]))
    
    # Put results as an xarray DataArray object.
    results = xr.Dataset(
        data_vars={"slopes": ((dim1, dim2), r[0, :, :]), 
                   "intercepts": ((dim1, dim2), r[1, :, :])},
        coords={dim1: getattr(data_array, dim1),
                dim2: getattr(data_array, dim2)}
    )

    # Maintain land mask coordinate into results.
    if "land_mask" in data_array.coords:
        results.coords["land_mask"] = data_array.land_mask
    
    return results

###############################################################################
def _theil_slopes_ufunc(
        y
    ):
    """
    Wrapper function for `scipy.stats.theilslopes` to be used in a vectorized 
    way in `theil_slopes_boosted` function.

    Parameters
    ----------
    y : numpy array
        One-dimensional data array.

    Returns
    -------
    results : numpy array
        Array containing the following results: (1) Theil slope, (2) Intercept
        of the Theil line, (3) Lower and (4) upper bounds of the confidence 
        interval on Theil slope.     
    """       

    # Dummy index in regression.
    x = np.arange(y.shape[0])
    
    # Just one not a number is sufficient to spoil calculations.
    if np.sum(np.isnan(y)) > 0:
        
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Output.
    else:

        slope, intercept, low_slope, upp_slope = stats.theilslopes(y=y, x=x)
        results = np.array([slope, intercept, low_slope, upp_slope])

        return results

###############################################################################
def theil_slopes_boosted(
        data_set, 
        var_code,
        dim="time"
    ):
    """
    Pixel-wise trends using Theil-Sen slope estimator. Vectorized 
    implementation of `_theil_slopes_ufunc`. For better performance, use 
    `data_set` input with chunked dask arrays.

    Parameters
    ----------
    data_set : xarray Dataset object
        Input data containting `time` dimension.         
    var_code : str
        Name of the variable inside `data_set` object.       
    dim : str, optional, default is "time"
        Trends will be calculated along this input core dimension.

    Returns
    -------
    results : xarray Dataset object
        Results of Theil-Sen estimator. This object contains variables for 
        slope (with lower and upper bounds of the confidence interval) and 
        intercept for each grid point.
    """    

    # Extract xarray DataArray object.
    data_array = getattr(data_set, var_code)
    
    # Standard data form.
    data_array = cdlearn.utils.organize_data(data_array)
   
    # Apply vectorized function.
    results_data_array = xr.apply_ufunc(
        _theil_slopes_ufunc, data_array,
        input_core_dims=[[dim]],
        output_core_dims=[["parameters"]],
        output_dtypes=["float32"],
        output_sizes={"parameters": 4},
        vectorize=True,
        dask="parallelized"
    )
    
    # Coordinates of this temporary dimension.
    results_data_array["parameters"] = [
        "slopes", "intercept", "lower_slope", "upper_slope"
    ]
    
    # Turn this xarray DataArray object into an xarray Dataset object deleting 
    # 'parameters' dimension. Now this Dataset has four variables: (1) Theil 
    # slope, (2) Intercept of the Theil line, (3) Lower and (4) upper bounds 
    # of the confidence interval on Theil slope. 
    results = results_data_array.to_dataset(dim="parameters")

    return results    