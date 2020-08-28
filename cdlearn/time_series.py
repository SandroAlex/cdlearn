"""
===============================================================================
Time series tools

Tools for retrieving and manipulating time series.
===============================================================================
"""

# Load packages.
import tensorflow as tf

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