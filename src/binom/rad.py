import numpy as np
import pvlib

"""
This module includes functions related to the configuration of BINOM's radiation-related parameters
"""

def add_ameanv(df, ameanv):
    """
    Add mean leaf absorptivity in the visible (PAR) band.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    ameanv : float
        Mean leaf absorptivity (0–1)

    Returns
    -------
    pandas.DataFrame
        Copy with 'ameanv' column added
    """
    df = df.copy()
    df["ameanv"] = ameanv
    return df

def add_rsoilv(df, rsoilv):
    """
    Add soil reflectivity in the visible (PAR) band.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    rsoilv : float
        Soil reflectivity (0–1)

    Returns
    -------
    pandas.DataFrame
        Copy with 'rsoilv' column added
    """
    df = df.copy()
    df["rsoilv"] = rsoilv
    return df

def add_clearsky_rad_stats(
    df,
    lat,
    lon,
    tz="America/Los_Angeles",
    altitude=0,
    model="ineichen",
    fvis=0.45
):
    """
    Add clear-sky radiation terms for BINOM using pvlib.

    Adds:
        - Srad_dir  : direct-beam irradiance on a horizontal surface [W m^-2]
        - Srad_diff : diffuse horizontal irradiance [W m^-2]
        - fvis      : fraction of incoming radiation in the visible band [0-1]

    Parameters
    ----------
    df : pandas.DataFrame
        Must be indexed by timezone-aware timestamps.
        Must contain either:
            - 'sza' in radians
          or
            - 'apparent_zenith' in degrees
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    tz : str
        Time zone string.
    altitude : float
        Site altitude in meters.
    model : str
        pvlib clear-sky model. Common options include 'ineichen',
        'haurwitz', and 'simplified_solis'.
    fvis : float
        Approximate visible-band fraction of incoming shortwave radiation.

    Returns
    -------
    pandas.DataFrame
        Copy of df with Srad_dir, Srad_diff, and fvis added.
    """
    out = df.copy()

    if out.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware.")

    if "sza" in out.columns:
        sza_rad = out["sza"].to_numpy()
        apparent_zenith_deg = np.rad2deg(sza_rad)
    elif "apparent_zenith" in out.columns:
        apparent_zenith_deg = out["apparent_zenith"].to_numpy()
        sza_rad = np.deg2rad(apparent_zenith_deg)
    else:
        raise ValueError("df must contain either 'sza' (radians) or 'apparent_zenith' (degrees).")

    site = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        tz=tz,
        altitude=altitude
    )

    # get_clearsky returns broadband ghi, dni, dhi
    cs = site.get_clearsky(
        times=out.index,
        model=model
    )

    # Convert direct normal irradiance to direct horizontal irradiance
    cosz = np.cos(sza_rad)
    cosz = np.clip(cosz, 0, None)

    out["Srad_dir"] = cs["dni"].to_numpy() * cosz
    out["Srad_diff"] = cs["dhi"].to_numpy()
    out["fvis"] = fvis

    return out