"""
===============================================================================
Utils

General utilities for cdlearn package.
===============================================================================
"""

# Physical constants related to Earth and its atmosphere. 
###############################################################################
radius = 6.3781e6 # Radius (m).
g0 = 9.8066       # Standard acceleration due to gravity (m/s2).

# Ancillary variables.
###############################################################################
months_labels = [       # Labels for months.
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

###############################################################################
months_labels_pt = [   # Labels for months in Portuguese.
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", 
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]

# Invariant data from ERA-INTERIM.
###############################################################################
folder_invariant = "/LFASGI/sandroal/data_sets/ERA_INTERIM/invariant/"

###############################################################################
dict_invariant = {
    "anor": "angle_of_sub_gridscale_orography.nc",
    "isor": "anisotropy_of_sub_gridscale_orography.nc",
    "z": "geopotential.nc",
    "cvh": "high_vegetation_cover.nc",
    "lsm": "land_sea_mask.nc",
    "cvl": "low_vegetation_cover.nc",
    "slor": "slope_of_sub_gridscale_orography.nc",
    "sdfor": "standard_deviation_of_filtered_subgrid_orography.nc",
    "sdor": "standard_deviation_of_orography.nc",
    "tvh": "type_of_high_vegetation.nc",
    "tvl": "type_of_low_vegetation.nc",
}

# Functions.
###############################################################################
def normalize_names(
        data_object
    ):
    """
    Get names of dimensions and return them as a tuple in the standard order: 
    (time, latitude, longitude)

    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    dims : tuple of str
        Names of dimensions.
    """

    dims = data_object.dims
    if "time" in dims:
        dim0 = "time"
    else:
        dim0 = None # If time dimension does not exist, then just return None.    

    if "lat" in dims:
        dim1 = "lat"
    elif "latitude" in dims:
        dim1 = "latitude"

    if "lon" in dims:
        dim2 = "lon"
    elif "longitude" in dims:
        dim2 = "longitude"

    return dim0, dim1, dim2    

###############################################################################
def organize_data(
        data_object
    ):
    """
    Put data in the standard form.
    
    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    data_object : xarray DataArray or Dataset object
        Data transposed for dimensions in the standard way and coordinates in 
        ascending order.
    """

    # Time, latitude, and longitude.
    dim0, dim1, dim2 = normalize_names(data_object) 

    # Just guarantee that time is the first dimension. 
    # Ascending ordered dimensions.
    data_object = data_object.transpose(dim0, dim1, dim2)
    data_object = data_object.sortby(dim0)
    data_object = data_object.sortby(dim1)
    data_object = data_object.sortby(dim2)

    return data_object

###############################################################################
def shift_longitude(
        data_object
    ):
    """
    Shift longitude values from [0º ... 360º] to [-180º ... 180º].
    Parameters
    ----------
    data_object : xarray DataArray or Dataset object
        Input data variable.

    Returns
    -------
    data_object : xarray DataArray or Dataset object
        Data with longitude axis shifted.
    """
    
    # Time, latitude, and longitude.
    _, _, dim2 = normalize_names(data_object)

    # Just do it!
    data_object = data_object.assign_coords(
        coords={dim2: (data_object[dim2] + 180) % 360 - 180}
    )
    data_object = data_object.sortby(data_object[dim2])

    return data_object