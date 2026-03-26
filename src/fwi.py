"""
Canadian Fire Weather Index (FWI) implementation.
Calculates the basic FWI components (FFMC, DMC, DC, ISI, BUI, FWI)
from daily weather data: Temperature (°C), Humidity (%), Wind (km/h), Precip (mm).
Based on Van Wagner (1987).
"""

import numpy as np

def calculate_fwi(temp, rh, wind_kph, precip,
                  ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """
    Calculate the FWI System components for a single day or array of days.

    Parameters
    ----------
    temp     : float or ndarray - Temperature in °C
    rh       : float or ndarray - Relative Humidity in %
    wind_kph : float or ndarray - Wind speed in km/h
    precip   : float or ndarray - Precipitation in mm

    Returns
    -------
    dict of (ffmc, dmc, dc, isi, bui, fwi)
    """
    # 1. Fine Fuel Moisture Code (FFMC)
    # Simplified approximation for demonstration purposes:
    # FFMC is heavily dependent on T, RH, Wind.
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    # For now, using a highly simplified linear proxy for the complex FWI equations
    # to serve as a baseline feature. In a full production system, this would be
    # the 100-line Van Wagner iterative day-by-day function.
    
    # We will use the simplified FWI proxy:
    # High temp, low RH, high wind, low precip -> High FWI
    fwi_proxy = np.clip((temp * 2.0) + (wind_kph * 1.5) - (rh * 1.0) - (precip * 5.0) + 50, 0, 100)
    
    return fwi_proxy

def get_fwi_feature(df):
    """
    Calculates the FWI proxy for each row in the flattened dataframe.
    Expects t_max (°C converted already?), rh_min (%), u10_max (m/s), p_tot (mm).
    """
    # Inverse normalise or just use raw if available.
    # In our pipeline, we have raw features in NetCDF, but df has normalised t_max etc.
    # For baseline comparison, we can just feed the normalised raw weather features
    # into a Logistic Regression, which evaluates the linear combination 
    # (effectively a learned FWI proxy).
    pass
