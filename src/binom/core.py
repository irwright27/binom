import numpy as np
import pandas as pd
from binom.path import pathlengthdistribution

"""
Core functions to run binom.
"""

def compute_binomial_ellipsoid(
    sr, #Row spacing (meters)
    #yup
    sza, # sun zenith angle (radians) check
    #rename
    psi, # sun azimuth relative to row orientation (radians)
    #yup
    lad, # Within-crown leaf area density (m2 m-3)  lad=LAI * sp * sr /(2/3*pi* r *r* H) ;
    #yup (why is it so high??? Is it because canopy height does not equal canopy volume?)
    ameanv, # Leaf absorptivity in the visible (PAR) band
    # Uses an average
    rsoilv, # Soil reflectivity in the visible (PAR) band
    # Uses an average
    Srad_dir,  # Direct-beam incoming radiation (W m-2)
    # Uses pvlib
    Srad_diff,  # Diffuse incoming radiation (W m-2)
    # Uses pvlib
    fvis, # Fraction incoming radiation in the visible part of the spectrum
    # Uses 0.45
    CrownVerticalRadius,# Crown vertical radius (meters)
    # yup
    z, # Distance between the center of the crown and the ground (m; z ≥ V)
    # yup
    wc, # Canopy width (meters)
    # yup
    sp, #Plant spacing (meters)
    # yup
    Gtheta, # fraction of leaf area projected in the direction of the sun
    # yup
    nrays,
    Nbins,
    shape="ellipsoid",   # <-- shape of the canopy
    Nz_diff=8, Nphi_diff=16,  # sampling for diffuse hemisphere
    return_diagnostics = False
):
    """
    Returns:
        Rc_PARdir: canopy absorbed direct radiation (Wm-2),
        Rs_PARdir: soil absorbed direct radiation (Wm-2),
        Rc_PARdiff: canopy absorbed diffuse radiation (Wm-2),
        Rs_PARdiff: soil absorbed diffuse radiation (Wm-2),


    References:
    - Bailey, B.N., Ponce de León, M.A., and Krayenhoff, E.S., 2020. One-dimensional models of radiation transfer in homogeneous canopies: A review, re-evaluation, and improved model. Geoscientific Model Development 13:4789:4808
    - Bailey, B.N. and Fu, K., 2022. The probability distribution of absorbed direct, diffuse, and scattered radiation in plant canopies with varying structure. Agricultural and Forest Meteorology, 322, p.109009.
    - Ponce de León, M.A., Alfieri, J.G., Prueger, J.H., Hipps, L., Kustas, W.P., Agam, N., Bambach, N., McElrone, A.J., Knipper, K., Roby, M.C. and Bailey, B.N., 2025.
      One-dimensional modeling of radiation absorption by vine canopies: evaluation of existing model assumptions, and development of an improved generalized model.
      Agricultural and Forest Meteorology, 373, p.110706 (https://doi.org/10.1016/j.agrformet.2025.110706)

    """

    r=wc/2.0
    CrownHeight=CrownVerticalRadius*2

    S0 = np.pi*r*r
    Stheta = np.pi * r ** 2 * np.sqrt(1 + (CrownVerticalRadius+z / (2 * r)) ** 2 * np.tan(sza) ** 2)



    s = (sr * (np.sin(psi) ** 2)) + (sp * (np.cos(psi) ** 2))
    s2 = s ** 2

    ##################### EDIT FOUR ##################################

    N_crown = CrownHeight * np.tan(sza) / s

    ##################### END EDIT FOUR ##################################

    IncPAR_dir = Srad_dir * fvis
    IncPAR_diff = Srad_diff * fvis

    dist = pathlengthdistribution(
        shape=shape,
        scale_x=wc,
        scale_y=sp,
        scale_z=CrownHeight,
        ray_zenith=sza,
        ray_azimuth=psi,
        nrays=int(nrays),
        bins=int(Nbins),
    )
    N = dist["hist"] / (np.sum(dist["hist"]))
    S = dist["bin_centers"]

    # Probability of intersecting a leaf within a prism during first and second order scattering
    #PlOne_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * ameanv * lad * S)))
    
    
    ##################### EDIT ONE ##################################
    # Probability of intersecting a leaf within a prism during first and second order scattering
    # REMOVED AMEANV FROM INSIDE OF EXPONENT. NOW THIS IS REPRESENTATIVE OF LEAF INTERSECTION
    PlOne_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * lad * S)))
    PlOne_PAR = PlOne_PAR*ameanv

    ##################### END EDIT ONE ##################################

    #PlTwo_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * 2.0 * ameanv * lad * S)))

    ##################### EDIT TWO ##################################
    # PLTWO SHOULD BE REPRESENTATIVE OF THE PROBABILITY OF LIGHT SCATTERING THEN BEING ABSORBED BY A LEAF
    # Therefore, we need a soil reflected PAR term first...

    # And soil_term_PAR needs to be dependent on the radiation that was scattered by the canopy (so therefore, we actually need Pc1_PAR First!!!)
    
    S0_over_s2 = np.where(s2 > 0, S0 / s2, 0.0) # unchanged

    # Recall N_crown was changed up in EDIT FOUR
    # Also, we've removed the azimuthally symetrical spacing adjuster ((s2 / (sr * sp)))
    # For justification, see Notability/Chapter 2/Binom investigation

    #Pc1_PAR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_PAR * S0_over_s2) ** N_crown) 
    Pc1_PAR = 1.0 - ((1.0 - PlOne_PAR * S0_over_s2) ** N_crown)

    spacing_adjuster = s2 / (sr * sp) # saving for diagnostic purposes

    # Why is PlOne_PAR a multiplier here? Wasn't that already factored into Pc1_PAR???
    soil_term_PAR = (1.0 - Pc1_PAR) * rsoilv * S0_over_s2# * PlOne_PAR

    # Now, PlTwo_PAR needs to be reliant also on the probability that light was scattered, right?
    # So we incorporate the PlOne_PAR and soil_term_PAR component
    # Lastly, took out the 2.0... Why is it there?
    PlTwo_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * ameanv * lad * S))) * (1 - PlOne_PAR + soil_term_PAR)


    ##################### END EDIT TWO ##################################

    #S0_over_s2 = np.where(s2 > 0, S0 / s2, 0.0)

    # Canopy-level probability of interception
    #Pc1_PAR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_PAR * S0_over_s2) ** N_crown)

    #Pc2_PAR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_PAR * S0_over_s2) ** N_crown) ######################################################### Take out the first "1 -""
    Pc2_PAR = (s2 / (sr * sp)) * (1.0 - PlTwo_PAR * S0_over_s2) ** N_crown
    

    #soil_term_PAR = (1.0 - Pc1_PAR) * rsoilv * S0_over_s2 * PlOne_PAR



    
    ##################### EDIT THREE ##################################

    # Direct radiation absorbed by the canopy
    
    Rc_PARdir = (Pc1_PAR) * IncPAR_dir

    # Rc_PARdir = (Pc2_PAR + soil_term_PAR) * IncPAR_dir #IncPAR_dir is all imcoming PAR! SHouldn't we apply the probability of absorbing a photon after scattering to the PAR that's already been scattered??

    ##################### END EDIT THREE ##################################







    ############################ ALL CHANGES DONE AT THIS POINT #####################################
    ################### NEXT STEP: WE NEED TO INCORPORATE A DIFFERENCE BETWEEN A LEAF REFLECTING A PHOTON AND TRANSMITTING A PHOTON (SEE MANUELS CHALK TALK)

    # save direct-beam versions for diagnostics
    s_direct = s
    s2_direct = s2
    S0_over_s2_direct = S0_over_s2

    # Direct radiation absorbed by the soil
    Rs_PARdir = IncPAR_dir * (1.0 - Pc1_PAR) * (1 - rsoilv)

    # ---- Diffuse sky part: integrate over hemisphere ----
    #  integration over zenith and azimuth:
    dtheta = (0.5 * np.pi) / Nz_diff
    dphi = (2.0 * np.pi) / Nphi_diff

    # Accumulators for per-crown interception averages (used in soil terms)
    PlOne_PAR_diff_tot = 0.0
    PlTwo_PAR_diff_tot = 0.0

    # Accumulators for canopy-level interception (crown overlap included)
    Pc1_PAR_diff = 0.0
    Pc2_PAR_diff = 0.0

    for i in range(Nz_diff):
        # midpoint zenith
        theta = (i + 0.5) * dtheta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(Nphi_diff):
            # midpoint azimuth
            phi = (j + 0.5) * dphi
            # directional weight for isotropic diffuse sky
            w_dir = (cos_theta * sin_theta * dtheta * dphi) / np.pi
            # pathlength distribution for this direction
            dist_d = pathlengthdistribution(
                shape=shape,
                scale_x=wc,
                scale_y=sp,
                scale_z=CrownHeight,
                ray_zenith=theta,
                ray_azimuth=phi,
                nrays=int(nrays // (Nz_diff * Nphi_diff)
                          if nrays >= Nz_diff * Nphi_diff
                          else max(100, int(nrays / (Nz_diff * Nphi_diff)))),
                bins=int(Nbins),
            )

            N_d = dist_d["hist"] / np.sum(dist_d["hist"])
            S_d = dist_d["bin_centers"]

            # projection function
            G_dir = Gtheta

            # per-crown interception fractions (per beam, per spectral band)
            PlOne_PAR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * ameanv * lad * S_d)))
            #PlTwo_PAR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * 2.0 * ameanv * lad * S_d)))


            ####### NOTE : PLTWO IS EDITED #######


            PlTwo_PAR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * 1.1 * ameanv * lad * S_d)))

            PlOne_PAR_diff_tot += w_dir * PlOne_PAR_d
            PlTwo_PAR_diff_tot += w_dir * PlTwo_PAR_d

            Sthetadiff = np.pi * r ** 2 * np.sqrt(1 + (CrownVerticalRadius + z / (2 * r)) ** 2 * np.tan(theta) ** 2)
            N_crowndiff = Sthetadiff / S0

            s = sr * np.sin(phi) ** 2 + sp * np.cos(phi) ** 2
            s2 = s ** 2

            S0_over_s2 = S0 / s2
            Pc1_PAR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_PAR_d * S0_over_s2) ** N_crowndiff)
            Pc2_PAR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_PAR_d * S0_over_s2) ** N_crowndiff)

    # ---- Diffuse contributions to soil and canopy ----
    Rs_PARdiff = IncPAR_diff * (1.0 - Pc1_PAR_diff) * (1.0 - rsoilv)

    soil_term_PAR_diff = (1.0 - Pc1_PAR_diff) * rsoilv * S0_over_s2 * PlOne_PAR_diff_tot

    Rc_PARdiff = (Pc2_PAR_diff + soil_term_PAR_diff) * IncPAR_diff

    if return_diagnostics:
        return {
            "Rc_PARdir": Rc_PARdir,
            "Rs_PARdir": Rs_PARdir,
            "Rc_PARdiff": Rc_PARdiff,
            "Rs_PARdiff": Rs_PARdiff,
            "Pl": PlOne_PAR,
            "PlTwo_PAR": PlTwo_PAR,
            "Pc": Pc1_PAR,
            "Pc2_PAR": Pc2_PAR,
            "soil_term_PAR": soil_term_PAR,
            "Pc1_PAR_diff": Pc1_PAR_diff,
            "Pc2_PAR_diff": Pc2_PAR_diff,
            "soil_term_PAR_diff": soil_term_PAR_diff,
            "s": s_direct,
            "s2": s2_direct,
            "S0_over_s2": S0_over_s2_direct,
            "S0": S0,
            "Stheta": Stheta,
            "N_crown": N_crown,
            "IncPAR_dir": IncPAR_dir,
            "PlOne_PAR * S0_over_s2": PlOne_PAR * S0_over_s2,
            "psii": psi, 
            "spacing_adjuster": spacing_adjuster,
            "sr*sp": sr * sp
        }

    else:
        return Rc_PARdir, Rs_PARdir, Rc_PARdiff, Rs_PARdiff

def binom_ts(df, model_prefs):
    """
    Run BINOM for every row in the dataframe and append outputs
    as new columns to a copy of the input dataframe, including canopy fAPAR.
    """

    nrays = model_prefs["nrays"]
    Nbins = model_prefs["Nbins"]
    shape = model_prefs.get("shape", "ellipsoid")
    Nz_diff = model_prefs.get("Nz_diff", 8)
    Nphi_diff = model_prefs.get("Nphi_diff", 16)

    out_df = df.copy()

    result_rows = []

    for row in out_df.itertuples():
        out = compute_binomial_ellipsoid(
            sr=row.sr,
            sza=row.sza,
            psi=row.psi,
            lad=row.lad,
            ameanv=row.ameanv,
            rsoilv=row.rsoilv,
            Srad_dir=row.Srad_dir,
            Srad_diff=row.Srad_diff,
            fvis=row.fvis,
            CrownVerticalRadius=row.CrownVerticalRadius,
            z=row.z,
            wc=row.wc,
            sp=row.sp,
            Gtheta=row.Gtheta,
            nrays=nrays,
            Nbins=Nbins,
            shape=shape,
            Nz_diff=Nz_diff,
            Nphi_diff=Nphi_diff,
            return_diagnostics=True,
        )

        if not isinstance(out, dict):
            raise TypeError(
                "compute_binomial_ellipsoid must return a dict when "
                "return_diagnostics=True"
            )

        result_rows.append(out)

    # Combine results
    results_only = pd.DataFrame(result_rows, index=out_df.index)
    out_df = pd.concat([out_df, results_only], axis=1)

    # --------------------------------------------------
    # Compute canopy fAPAR
    # --------------------------------------------------

    # Removed diffuse components of fAPAR for now...

    absorbed_canopy = out_df["Rc_PARdir"] # + out_df["Rc_PARdiff"]
    incoming_par = (out_df["IncPAR_dir"]) # + out_df["Srad_diff"]) * out_df["fvis"]

    # Avoid divide-by-zero
    out_df["fAPAR_direct"] = np.where(
        incoming_par > 0,
        absorbed_canopy / incoming_par,
        np.nan
    )

    # Optional: flag unphysical values
    out_df["fAPAR_flag"] = out_df["fAPAR_direct"] > 1.0

    return out_df
