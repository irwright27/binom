import os
import numpy as np
from numpy import sqrt, sin, arcsin, cos, arccos, exp, pi, linspace, ceil

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