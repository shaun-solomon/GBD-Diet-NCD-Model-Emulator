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
    NOTE - For this analytical scenario this function does not yield a single Joint PAF for all 15 dietary risks for a specific outcome 
    MF (mediation matrix) - A modified MF is called before using this function (either calculate_MF_NJ() or calculate_MF_J())
    - If calculate_MF_NJ() is called then Joint PAF collapses to the non-joint (without considering mediation effects) individual PAF for that risk 
    - If calculate_MF_J() is called the Joint PAF collapses to the joint (considering mediation effects) individual PAF for that risk 
    """
    PAF_J = 1 - np.prod(1 - PAFs * np.prod(1 - MF[:, :, o], axis=1))
    return PAF_J


def change_joint_PAFs_per_disease(PAFs, PAFs_shifted, MF, o): 
    '''
    Calculates difference between shifted PAF and original PAF
    PAFs: Vector with PAFs for different risks 
    PAFs_shifted : Vector with shifted PAFs corresponding to a unilateral shift in intake
    MF : mediation matrix
    o : corresponding disease index
    Returns: Change in joint or non-joint PAFs (depending on the mediation matrix used)
   '''
    PAF_J_change = (1 - np.prod(1 - PAFs_shifted * np.prod(1 - MF[:, :, o], axis=1))) - (1 - np.prod(1 - PAFs * np.prod(1 - MF[:, :, o], axis=1)))
    return PAF_J_change


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
    
# Creating another function to calculate shifted PAF values 
def full_calculation_shift(scenario, time, country, disease, age, gender, risk, TMREL_df, risks_df, distribution_df, rf_df_morb,
                     rf_df_mort, shift_df , morb_mort='Both', run=1):
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
    shift_df (dataframe): dataframe containing h values for each combination of risk, age, sex, and country
    Returns: PAF value
    """
    # get TMREL and unit from the risk factors df
    TMREL = TMREL_df.loc[risk, str(run)]
    unit = risks_df.loc[(risk, disease, morb_mort), 'Units']
    low = risks_df.loc[(risk, disease, morb_mort), 'Low']
    # load the shift applied to this risk's exposure (scenario/time/country/age/sex/risk-specific)
    h = shift_df.loc[(scenario, time, country, age, gender, risk), str(run)]

    # get the sample
    x_array = distribution_df.loc[(risk, age, gender), :].to_numpy()

    # get risk factor from the dataframe, only the float before the brackets is used
    if morb_mort == 'Morbidity':
        rf = rf_df_morb.loc[(risk, disease, age), str(run)]
    else:
        rf = rf_df_mort.loc[(risk, disease, age), str(run)]

    # Create the histogram 
    bin_edges = np.linspace(min(x_array), max(x_array), 101) # bin edges are assigned to 101, number of bins does not change from the function full_calculation 
    hist, _ = np.histogram(x_array, bins=bin_edges, density=True)
    
    # calculate risk factors per bin
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    centers_shifted = h * np.ones_like(centers) + centers #apply shift h 
    rr_shifted = np.array([])
    for center in centers_shifted:
        rr_shifted = np.append(rr_shifted, calculate_rr(center, TMREL, rf, unit, low))
    
    # calculate shifted PAFs
    rr_upper_shifted = rr_shifted - 1
    nominator_integral = sp.integrate.simpson(rr_upper_shifted * hist, x=centers)
    denominator_integral = sp.integrate.simpson(rr_shifted * hist, x=centers)
    
    PAF_shifted = nominator_integral / denominator_integral
    if np.isnan(PAF_shifted):
        print('PAF_shifted is none')
        exit()

    return PAF_shifted
    
# Creating a function for proportional PAFs 
def calculate_PJ_PAFs(PAFs_J, PAF_J):
    """
    This function decomposes the total Joint PAF into proportional, risk specific contributions. 
    The resulting PAFs sum to the overall Joint PAF
    """
     
    factor = np.sum(PAFs_J)
    
    # Check for zero or NaN factors
    if factor == 0 or np.isnan(factor):
        return np.zeros_like(PAFs_J)
    
    # Calculate the proportional PAF for each risk
    PAF_prop = PAFs_J * (PAF_J / factor)
    
    return PAF_prop

