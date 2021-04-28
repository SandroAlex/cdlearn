"""
===============================================================================
Time series tools

Tools for retrieving and manipulating time series.
===============================================================================
"""

# Load packages.
import numpy as np
import xarray as xr
import tensorflow as tf

from importlib import reload

# My modules.
import cdlearn.statistics

# Incorporate ongoing changes.
reload(cdlearn.statistics)

# Functions.
###############################################################################
def univariate_auto_features(
        time_series,
        window_size,
    ):
    """
    Create sequence windows of data from a single time series.

    Parameters
    ----------
    time_series : np.array
        Target univariate time series.
    window_size : int       
        Maximum shift to be used in features.
 
    Returns
    -------
    X : 2D np.ndarray
        The dataset with windowed values of the target.
    y : np.array 
        Target values one step ahead.
    """  

    # Size of the time series.
    n = time_series.shape[0]    
    
    # Reshape data in order to be used in data generator.
    data = time_series.reshape((-1, 1))
    targets = data.copy()
    
    # Utility class for generating batches of temporal data.
    dg = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=data,
        targets=targets,
        length=int(window_size),
        batch_size=n
    )
    
    # Just one batch of data. The data generator instance has length one.
    Xgen, ygen = dg[0]
    
    # Reshape data in order to fit sklearn's Classes.
    # X: Instances (rows) and features (columns).
    X = Xgen.reshape((Xgen.shape[0], -1))
    y = ygen.reshape((ygen.shape[0], ))
    
    return X, y

###############################################################################
def make_sequences(
        features, 
        target, 
        window_size
    ):
    """
    Create sequence windows of data from temporal features and target.

    Parameters
    ----------
    features : 2D np.array
        Temporal data matrix.
    target : np.array    
        Target univariate time series.
    window_size : int       
        Maximum shift to be used in features.
 
    Returns
    -------
    X : 2D np.ndarray
        The dataset with windowed values of the features.
    y : np.array 
        Target values one step ahead.
    """
    # Size of the time series.
    n = features.shape[0]  
    
    # Utility class for generating batches of temporal data.
    dg = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=features,
        targets=target,
        length=int(window_size),
        batch_size=n
    )
    
    # Just one batch of data. The data generator instance has length one.
    Xgen, ygen = dg[0]
    
    # Reshape data in order to fit sklearn's Classes.
    # X: Instances (rows) and features (columns).
    X = Xgen.reshape((Xgen.shape[0], -1))
    y = ygen.reshape((ygen.shape[0], ))
    
    return X, y    

###############################################################################
def permute_years_15day(
        time_index, 
        verbose=False
    ):
    """
    Permute years without altering months neither days order in the time index.

    Parameters
    ----------
    time_index : xarray DataArray object
        Temporal data indexes.
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.
 
    Returns
    -------
    time_index_shuffled : xarray DataArray object
        Shuffled temporal data indexes.
    """

    # This copy will be reduced at each iteration in the above loop.
    time_aux = time_index.dt.strftime("%Y-%m-%d").values
    
    # This order must be respected.
    months_and_days = [
        month.zfill(2) + "-" + day.zfill(2) 
        for month, day in zip(
            time_index.dt.month.values.astype(np.str),
            time_index.dt.day.values.astype(np.str)
        ) 
    ] 
        
    # Years without repetition.
    years_unique = np.unique(time_index.dt.year).astype(str)
    
    # Final time index.
    results = []
    
    # Build permuted time index.
    for run in range(time_index.size):
 
        month_and_day = months_and_days[run]
        keep_searching = True
        
        # Ok! you can go!
        while keep_searching:
        
            year = np.random.choice(years_unique)
            time_result = year + "-" + month_and_day
        
            if time_result in time_aux:
                
                results.append(time_result)
                time_aux = np.delete(time_aux, np.where(time_aux==time_result))
                keep_searching = False
                
                if verbose:
                    print(time_result, " OK!")
        
            else:
                keep_searching = True

    # As numpy array.
    results = np.array(results).astype(np.datetime64)
    
    # Cumbersome, but in agreement with input.
    time_index_shuffled = xr.DataArray(
        data=results, dims=["time"], coords={"time": results}
    )
    
    return time_index_shuffled

###############################################################################
def permute_years_monthly(
        time_index, 
        verbose=False
    ):
    """
    Permute years without altering months order in the time index.

    Parameters
    ----------
    time_index : xarray DataArray object
        Temporal data indexes.
    verbose : bool, optional, default is False
        If True, then prints a progress bar for loop over spatial grid points.
 
    Returns
    -------
    time_index_shuffled : xarray DataArray object
        Shuffled temporal data indexes.    
    """

    # This copy will be reduced at each iteration in the above loop.
    time_aux = time_index.dt.strftime("%Y-%m").values
    
    # This order must be respected.
    months = time_index.dt.month.values.astype(str)
    
    # Years without repetition.
    years_unique = np.unique(time_index.dt.year).astype(str)
    
    # Final time index.
    results = []
    
    # Build permuted time index.
    for run in range(time_index.size):
 
        month = months[run]
        keep_searching = True
        
        # Ok! you can go!
        while keep_searching:
        
            year = np.random.choice(years_unique)
            time_result = year + "-" + month.zfill(2)
        
            if time_result in time_aux:
                
                results.append(time_result)
                time_aux = np.delete(time_aux, np.where(time_aux==time_result))
                keep_searching = False
                
                if verbose:
                    print(time_result, " OK!")
        
            else:
                keep_searching = True

    # As numpy array.
    results = np.array(results).astype(np.datetime64)
    
    # Cumbersome, but in agreement with input.
    time_index_shuffled = xr.DataArray(
        data=results, dims=["time"], coords={"time": results}
    )
    
    return time_index_shuffled    

###############################################################################
def shuffle_data_by_years(
        data_set, 
        var_code,
        time_step="15day"
    ):
    """
    Permute years without altering order of months and/or days in the input 
    data.

    Parameters
    ----------
    data_set : xarray Dataset object
        Data container.
    var_code : str
        Name of the variable inside data container.
    time_step : str, "15day" or "monthly"
        Time step for data.    
 
    Returns
    -------
    data_set_shuffled : xarray Dataset object
        Data container with shuffled temporal indexes.     
    """
    
    # New time index.
    if time_step == "monthly":
        time_index_shuffled = permute_years_monthly(data_set.time)
    
    elif time_step == "15day":
        time_index_shuffled = permute_years_15day(data_set.time)

    else:
        raise Exception("Time step error: " + \
                        "available options are (1) '15day' and (2) 'monthly'")
        
    # New time-shuffled data
    data_set_shuffled = data_set.copy(deep=True)
    data_set_shuffled["time"] = time_index_shuffled
    data_set_shuffled = data_set_shuffled.sortby("time")
    
    return data_set_shuffled    