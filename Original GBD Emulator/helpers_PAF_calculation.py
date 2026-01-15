import numpy as np
import scipy as sp


def calculate_rr(x, TMREL, rf, unit, low=True):
    """
    This function calculates the actual risk factors, depending on
    TMREL: the TMREL(per risk-disease pair),
    rf: the rf (per age group and risk-disease pair) and
    unit: the unit in which the consumption (TRMEL) is measured
    x: specific consumption level
    """
    if low:
        if x >= TMREL:
            return 1
        else:
            return rf ** ((TMREL - x) / unit)
    else:
        if x <= TMREL:
            return 1
        else:
            return rf ** ((x - TMREL) / unit)


def calculate_PAF_per_disease(PAFs, MF, o):
    """
    this function is for calculating the PAFs considering possible overlaps between different risk factors with the
    mediation matrix
    PAF: vector with PAFs for different risks
    MF: mediation matrix
    o: corresponding disease index
    """
    PAF_J = 1 - np.prod(1 - PAFs * np.prod(1 - MF[:, :, o], axis=1))
    return PAF_J


def full_calculation(risk, disease, age, gender, TMREL_df, risks_df, distribution_df, rf_df_morb,
                     rf_df_mort, morb_mort='Both', run=1):
    """
    calculates the PAF value depending on
    risk (str): Risk
    disease (str): Disease
    gender (str): Sex specification
    morb_mort (str): specifies whether the PAF relates to Morbidity or Mortality
    run (int): specifies the number of the specific run, of the code (between 0 and 999)
    TMREL_df (datdframe): Containing the TMREL values
    risk_df (datdframe): Contains the specifcation of the unit and risk-high-or-low-inidcator
    distribution_df (dataframe): contains the values of the distribution for each age, gender, risk combination
    rf_df_morb (dataframe): rf of run for morbidity
    rf_df_mort (dataframe): rf of run for mortality
    Returns: PAF value
    """
    # get TMREL and unit from the risk factors df
    TMREL = TMREL_df.loc[risk, str(run)]
    unit = risks_df.loc[(risk, disease, morb_mort), 'Units']
    low = risks_df.loc[(risk, disease, morb_mort), 'Low']

    # get the sample
    x_array = distribution_df.loc[(risk, age, gender), :].to_numpy()

    # get risk factor from the dataframe, only the float before the brackets is used
    if morb_mort == 'Morbidity':
        rf = rf_df_morb.loc[(risk, disease, age), str(run)]
    else:
        rf = rf_df_mort.loc[(risk, disease, age), str(run)]

    # create the histogram
    hist, bin_edges = np.histogram(x_array, bins=100, density=True)

    # calculate risk factors per bin
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    rr = np.array([])
    for center in centers:
        rr = np.append(rr, calculate_rr(center, TMREL, rf, unit, low))

    # calculate PAFs
    rr_upper = rr - 1
    nominator_integral = sp.integrate.simpson(rr_upper * hist, x=centers)
    denominator_integral = sp.integrate.simpson(rr * hist, x=centers)

    PAF = nominator_integral / denominator_integral
    if np.isnan(PAF):
        print('PAF is none')
        exit()

    return PAF
