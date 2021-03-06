"""
===============================================================================
Climate data learn

Python package intended to manipulate and analyze climate data in the framework
of Granger causality and machine learning.
===============================================================================
"""

# Load packages.
import os
from importlib import reload

# Upper folder for this module. Absolute path.
module_path = os.path.dirname(os.path.dirname(__file__))

# My modules. 
from . import maps
from . import pixels
from . import statistics
from . import time_series
from . import utils 

# Incorporate ongoing changes.
reload(maps)
reload(pixels)
reload(statistics)
reload(time_series)
reload(utils)

# Information.
__version__ = "0.1.0"
__author__ = "Alex Araujo <alex.fate2000@gmail.com>"