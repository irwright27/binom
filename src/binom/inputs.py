from binom.geometry import (get_spos,
                            add_psi,
                            add_canopy_geometry,
                            add_lad,
                            add_Gtheta)
from binom.rad import (add_ameanv,
                       add_clearsky_rad_stats,
                       add_rsoilv)


"""
This module contains the function to prepare the dataframe necessary to run binom as a time series using binom_ts
The prepare_binom_inputs function orchestrates all time based geometry functions into one simple wrapper
This function should be fed a dictionary with all model parameters (see READ_ME)
"""

def prepare_binom_inputs(model_params):

    lat = model_params["lat"]
    lon = model_params["lon"]
    date = model_params["date"]
    freq = model_params["freq"]
    row_azimuth = model_params["row_azimuth"]
    ameanv = model_params["ameanv"]
    rsoilv = model_params["rsoilv"]
    lai = model_params["lai"]
    Gtheta_model = model_params["Gtheta_model"]

    canopy_params = {
        "CrownVerticalRadius": model_params["CrownVerticalRadius"],
        "z": model_params["z"],
        "wc": model_params["wc"],
        "sp": model_params["sp"],
        "sr": model_params["sr"]
    }


    df = get_spos(lat=lat, lon=lon, date=date, freq=freq)
    df = add_psi(df, row_azimuth)
    df = add_ameanv(df, ameanv)
    df = add_rsoilv(df, rsoilv)
    df = add_clearsky_rad_stats(df, lat=lat, lon=lon)
    df = add_canopy_geometry(df, canopy_params)
    df = add_lad(df, lai, canopy_params)
    df = add_Gtheta(df, model=Gtheta_model)
    
    return df