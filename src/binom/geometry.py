from numpy import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

"""
This module contains all functions related to the configuration/processing of BINOM's geometric components
Path length related functions are not included in this module. See path.py
Note: LAD processing is also included here, as it is considered a geometric parameter
"""

def get_spos(
    lat,
    lon,
    date,
    tz="America/Los_Angeles",
    altitude=0,
    freq="5min",
    daylight_only=True,
    include_vectors=False
):
    """
    Generate solar position dataframe for a single day
    with BINOM-compatible parameter names.

    Returns
    -------
    df : pandas.DataFrame
        Indexed by time, includes:
            - sza (rad)
            - azimuth (deg)
            - optional direction vectors (dx, dy, dz)
    """

    # --------------------------------------------------
    # Time index
    # --------------------------------------------------
    times = pd.date_range(
        start=f"{date} 00:00:00",
        end=f"{date} 23:55:00",
        freq=freq,
        tz=tz
    )

    # --------------------------------------------------
    # Location
    # --------------------------------------------------
    site = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        tz=tz,
        altitude=altitude
    )

    # --------------------------------------------------
    # Solar position
    # --------------------------------------------------
    solpos = site.get_solarposition(times)

    df = solpos[
        ["apparent_zenith", "azimuth", "apparent_elevation"]
    ].copy()

    # --------------------------------------------------
    # Daylight filter
    # --------------------------------------------------
    if daylight_only:
        df = df[df["apparent_elevation"] > 0].copy()

    # --------------------------------------------------
    # Rename + compute BINOM variables
    # --------------------------------------------------
    df["sza"] = np.deg2rad(df["apparent_zenith"])  # <-- key rename

    # --------------------------------------------------
    # Optional: sun direction vectors (for BINOM)
    # Convention: x=east, y=north, z=up
    # Rays pointing FROM sun → ground
    # --------------------------------------------------
    if include_vectors:
        zen = df["sza"].values
        azi = np.deg2rad(df["azimuth"].values)

        sx = np.sin(zen) * np.sin(azi)
        sy = np.sin(zen) * np.cos(azi)
        sz = np.cos(zen)

        df["dx"] = -sx
        df["dy"] = -sy
        df["dz"] = -sz

    return df

def wrap_angle_pi(angle_rad):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def add_psi(df, row_azimuth_deg):
    """
    Add psi (sun azimuth relative to row orientation) to dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain column 'azimuth' in degrees (from pvlib)
    row_azimuth_deg : float
        Row orientation in degrees clockwise from north

    Returns
    -------
    pandas.DataFrame
        Copy with 'psi' column added (radians)
    """
    df = df.copy()

    # Convert to radians
    sun_azimuth_rad = np.deg2rad(df["azimuth"].values)
    row_azimuth_rad = np.deg2rad(row_azimuth_deg)

    # Compute relative angle
    psi = wrap_angle_pi(sun_azimuth_rad - row_azimuth_rad)

    df["psi"] = psi

    return df

def add_Gtheta(df, model="spherical", chi=None):
    """
    Add G(theta) to dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'sza' (radians)
    model : str
        "spherical" (default) or "ellipsoidal"
    chi : float or None
        Shape parameter for ellipsoidal distribution

    Returns
    -------
    pandas.DataFrame
        Copy with 'Gtheta' column added
    """

    df = df.copy()

    if "sza" not in df.columns:
        raise ValueError("DataFrame must contain 'sza' (radians)")

    if model == "spherical":
        df["Gtheta"] = 0.5

    elif model == "ellipsoidal":
        if chi is None:
            raise ValueError("chi must be provided for ellipsoidal model")

        theta = df["sza"].values

        # Campbell (1986) approximation
        # This is a commonly used formulation
        x = chi

        G = np.sqrt(x**2 + np.tan(theta)**2) / (x + 1.774 * (x + 1.182)**-0.733)

        df["Gtheta"] = G

    else:
        raise ValueError(f"Unknown model: {model}")

    return df

def add_canopy_geometry(df, canopy_params):
    """
    Add canopy geometry parameters to dataframe from a dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    canopy_params : dict
        Must contain:
            - CrownVerticalRadius
            - z
            - wc
            - sp
            - sr

    Returns
    -------
    pandas.DataFrame
        Copy with canopy geometry columns added
    """

    required_keys = ["CrownVerticalRadius", "z", "wc", "sp", "sr"]

    # Check inputs
    missing = [k for k in required_keys if k not in canopy_params]
    if missing:
        raise ValueError(f"Missing keys in canopy_params: {missing}")

    df = df.copy()

    for key in required_keys:
        df[key] = canopy_params[key]

    return df

def compute_lad(LAI, canopy_params):
    """
    Compute within-crown leaf area density (LAD) using LAI and canopy geometry.

    Parameters
    ----------
    LAI : float
        Leaf area index [m2 m-2]
    canopy_params : dict
        Must contain:
            - sp : plant spacing [m]
            - sr : row spacing [m]
            - CrownVerticalRadius : vertical crown radius [m]
            - wc : canopy width [m]

    Returns
    -------
    float
        Leaf area density [m2 m-3]
    """

    required_keys = ["sp", "sr", "CrownVerticalRadius", "wc"]
    missing = [k for k in required_keys if k not in canopy_params]
    if missing:
        raise ValueError(f"Missing keys in canopy_params: {missing}")

    sp = canopy_params["sp"]
    sr = canopy_params["sr"]
    CrownVerticalRadius = canopy_params["CrownVerticalRadius"]
    wc = canopy_params["wc"]

    # Geometry
    r = wc / 2.0
    H = 2.0 * CrownVerticalRadius

    crown_volume = (2.0 / 3.0) * np.pi * r**2 * H

    if crown_volume <= 0:
        raise ValueError("Computed crown volume must be > 0.")

    lad = (LAI * sp * sr) / crown_volume

    return lad

def add_lad(df, LAI, canopy_params):
    """
    Add LAD to dataframe using LAI and canopy geometry dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    LAI : float
        Leaf area index [m2 m-2]
    canopy_params : dict
        Must contain:
            - sp
            - sr
            - CrownVerticalRadius
            - wc

    Returns
    -------
    pandas.DataFrame
        Copy with 'lad' and 'sr' columns added
    """

    df = df.copy()

    lad_value = compute_lad(LAI, canopy_params)

    df["lad"] = lad_value
    df["sr"] = canopy_params["sr"]

    return df

def intersectBBox(ox, oy, oz, dx, dy, dz, sizex, sizey, sizez):

    # Intersection code below is adapted from Suffern (2007) Listing 19.1

    x0 = -0.5*sizex
    x1 = 0.5*sizex
    y0 = -0.5 * sizey
    y1 = 0.5 * sizey
    z0 = -1e-6
    z1 = sizez

    if dx == 0:
        a = 1e6
    else:
        a = 1.0 / dx
    if a >= 0:
        tx_min = (x0 - ox) * a
        tx_max = (x1 - ox) * a
    else:
        tx_min = (x1 - ox) * a
        tx_max = (x0 - ox) * a

    if dy == 0:
        b = 1e6
    else:
        b = 1.0 / dy
    if b >= 0:
        ty_min = (y0 - oy) * b
        ty_max = (y1 - oy) * b
    else:
        ty_min = (y1 - oy) * b
        ty_max = (y0 - oy) * b

    if dz == 0:
        c = 1e6
    else:
        c = 1.0 / dz
    if c >= 0:
        tz_min = (z0 - oz) * c
        tz_max = (z1 - oz) * c
    else:
        tz_min = (z1 - oz) * c
        tz_max = (z0 - oz) * c

    # find largest entering t value

    if tx_min > ty_min:
        t0 = tx_min
    else:
        t0 = ty_min

    if tz_min > t0:
        t0 = tz_min

    # find smallest exiting t value

    if tx_max < ty_max:
        t1 = tx_max
    else:
        t1 = ty_max

    if tz_max < t1:
        t1 = tz_max

    if t0 < t1 and t1 > 1e-6:
        if t0 > 1e-6:
            dr = t1-t0
        else:
            dr = t1
    else:
        dr = 0

    xe = ox + t1 * dx
    ye = oy + t1 * dy
    ze = oz + t1 * dz

    if dr == 0:
        return 0, None, None, None  # signal no intersection
    return dr, xe, ye, ze

def intersectEllipsoid(ox, oy, oz, dx, dy, dz, sizex, sizey, sizez):

    tempx = ox/sizex
    tempy = oy/sizey
    tempz = (oz - 0.5)/sizez

    dx = dx/sizex
    dy = dy/sizey
    dz = dz/sizez

    a = dx*dx + dy*dy + dz*dz

    b = 2.0 * (tempx*dx+tempy*dy+tempz*dz)

    c = (tempx*tempx+tempy*tempy+tempz*tempz) - 0.5*0.5
    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return 0
    else:
        e = sqrt(disc)
        denom = 2.0 * a

        t_small = (-b - e) / denom  # smaller root
        t_big = (-b + e) / denom  # larger root

        if t_small > 1e-6 or t_big > 1e-6:
            dr = abs(t_big-t_small)
            return dr

    return 0