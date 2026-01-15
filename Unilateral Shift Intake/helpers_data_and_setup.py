import pandas as pd
import numpy as np
from Setup_file import min_max_path, distrbution_weights_path
from helpers import index_dict


def load_input_files():
    """
    Loads the minimum and maximum of each distribution and the file with the weights for each candidate distribution.
    returns: relative minimum and maximum dataframe,
             distribution weights dataframe
    """
    # loading the minimum and maximum files
    minmax_bounds_df = pd.read_csv(min_max_path, index_col=[0, 1, 2, 3])
    minmax_bounds_df = minmax_bounds_df.swaplevel(0, 1)
    minmax_bounds_df.sort_index(inplace=True)

    # loading the distrbution weights
    distribution_weights_df = pd.read_csv(distrbution_weights_path,
                                          delimiter=';',
                                          usecols=['Risk', 'exp', 'gamma', 'llogis', 'gumbel', 'weibull', 'lnorm',
                                                   'norm', 'betasr', 'mgamma', 'mgumbel', 'invgamma', 'invweibull'],
                                          index_col=[0])

    return minmax_bounds_df, distribution_weights_df


def load_mean_and_std(path, country):
    """
    Loads the mean and standard deviation files (for selected country)
    path (str): relative path where the files are stored
    country (int/str): indicator specifying the country (either name or M49)
    Returns mean values dataframe, std values dataframe (index: risk factor, age, gender, columns: range(num_runs))
    """
    mean_values_df = pd.read_csv(path.format('Mean', country), index_col=[0, 1, 2], header=0)
    mean_values_df.sort_index(inplace=True)
    sd_values_df = pd.read_csv(path.format('Std', country), index_col=[0, 1, 2], header=0)
    sd_values_df.sort_index(inplace=True)
    return mean_values_df, sd_values_df


def load_total_YLDs_YLLs(path):
    """
    Loads total YLD (Years Lived with Disability) and YLL (Years of Life Lost) dataframes from CSV files.
    path (str): The relative path where the files are stored.
    Returns: total_YLD_df, total_YLL_df dataframes indexed by location, cause, age, and sex.
    """
    total_YLD_df = pd.read_csv(path.format('YLD'), usecols=['location', 'sex', 'cause', 'age', 'val'],
                               index_col=[0, 1, 2, 3])
    total_YLD_df = total_YLD_df.reorder_levels(['location', 'cause', 'age', 'sex']).sort_index()
    total_YLL_df = pd.read_csv(path.format('YLL'), usecols=['location', 'sex', 'cause', 'age', 'val'],
                               index_col=[0, 1, 2, 3])
    total_YLL_df = total_YLL_df.reorder_levels(['location', 'cause', 'age', 'sex']).sort_index()
    return total_YLD_df, total_YLL_df


def create_full_min_max_df(min_max_df, risks, age_groups, genders):
    """
    Creates a new dataframe with multi-level indices based on 'risks', 'age_groups', and 'genders'.
    min_max_df: The original dataframe with minimum and maximum values.
    risks (list): List of risk factors.
    age_groups (list): List of age groups.
    genders (list): List of genders.
    Returns: A new dataframe with multi-level indices and 'xmin', 'xmax' columns.
    """
    min_max_new = pd.DataFrame(index=pd.MultiIndex.from_product([genders, risks, age_groups]),
                               columns=['xmin', 'xmax'])
    for gender in genders:
            min_max_new.loc[gender, 'xmin':'xmax'] = min_max_df.loc[:, 'xmin':'xmax'].to_numpy()
    min_max_new = min_max_new.swaplevel(0, 1).swaplevel(1, 2)
    min_max_new.sort_index(inplace=True)
    return min_max_new


def calculate_mediation_matrix():
    '''
    Calculates the mediation matrix, mediation values are now hardcoded and not loaded anymore.
    Returns: Mediation matrix (dim 15x15x13)
    '''
    risks_idx_dict = {}
    diseases_idx_dict = {}

    for idx, risk in enumerate(index_dict['risks']):
        risks_idx_dict[risk] = idx

    for idx, disease in enumerate(index_dict['diseases']):
        diseases_idx_dict[disease] = idx

    # create mediation matrix
    MF = np.zeros((len(index_dict['risks']), len(index_dict['risks']), len(index_dict['diseases'])))

    # triples which are correlated, they all have a mediation factor of 1
    triples = [('Diet low in milk', 'Diet low in calcium', 'Colon and rectum cancer'),
               ('Diet low in fiber', 'Diet low in fruits', 'Ischemic heart disease'),
               ('Diet low in fiber', 'Diet low in vegetables', 'Ischemic heart disease'),
               ('Diet low in fiber', 'Diet low in whole grains', 'Ischemic heart disease')]

    # loop over all triples
    for triple in triples:
        risk_idx_1 = risks_idx_dict[triple[0]]
        risk_idx_2 = risks_idx_dict[triple[1]]
        disease_idx = diseases_idx_dict[triple[2]]

        MF[risk_idx_1, risk_idx_2, disease_idx] = 1

    return MF

def calculate_MF_NJ(risk_factor):
    '''
    This function is used to create a modified mediation matrix to calculate non-joint PAFs
    Specified conditions -  Create a matrix with all 1s, except for the row containing the risk_factor that needs to be isolated 
    This collapses PAF_J (combined PAF that considers the overlaps across all 15 risks) to PAF_nj (individual PAF for that risk that DOES NOT consider mediation)
    Parameters - risk_factor - the name of the risk factor to be isolated. 
    '''
    
    risks_idx_dict = {}
    
    for idx, risk in enumerate(index_dict['risks']):
        risks_idx_dict[risk] = idx

    # create a 15x15x13 mediation matrix with all 1s
    MF_NJ = np.ones((len(index_dict['risks']), len(index_dict['risks']), len(index_dict['diseases'])))
    
    # setting the row containing the risk_factor to all 0s
    risk_idx_1 = risks_idx_dict[risk_factor]
    
    # set the row in all matrices to be zero 
    MF_NJ[risk_idx_1, :, :] = 0
    
    return MF_NJ
      
def calculate_MF_J(MF, risk_factor):
    '''
    This function is used to create a modified mediation matrix to calculate joint PAFs
    Specified conditions - 
    - The modified matrix will contain all 1s, except for the row containing the risk_factor that needs to be isolated
    - The row containing the risk_factor will be set to the same value as the original mediation matrix
    This collapses PAF_J (combined PAF that considers the overlaps across all 15 risks) to PAF_j (individual PAF for that risk that CONSIDERS mediation)
    Parameters - 
    MF - the original mediation matrix
    risk_factor - the name of the risk factor to be isolated. 
    '''
    
    risks_idx_dict = {}
    
    for idx, risk in enumerate(index_dict['risks']):
        risks_idx_dict[risk] = idx

    # create a 15x15x13 mediation matrix with all 1s
    MF_J = np.ones((len(index_dict['risks']), len(index_dict['risks']), len(index_dict['diseases'])))
    
    # setting the row containing the risk_factor to all 0s
    risk_idx_1 = risks_idx_dict[risk_factor]
    
    # set the row in all matrices to be zero 
    MF_J[risk_idx_1, :, :] = MF[risk_idx_1, :, :]
    
    return MF_J

def convert_to_dataframe(DALYs_array, index_dict, diseases, age_groups, genders, risks, countries, num_scenarios, num_times, num_countries, num_diseases, num_ages, num_genders, num_risks, num_runs):
    """
    - This function converts the multi-dimensional DALYs output array into a long format pandas df.
    - It iterates over scenarios, time points, countries, diseases, age groups, genders, risks, and the runs and maps each value in the DALYs array to its corresponding metadata labels. 
    - Output: A tidied df 
    """
    scenarios_dalys, years_dalys, countries_dalys, diseases_dalys, genders_dalys, risks_dalys, age_groups_dalys, run_values_dalys = [], [], [], [], [], [], [], []

    for scenario_idx in range(num_scenarios):
        for year_idx in range(num_times):
            for country_idx in range(num_countries):
                for disease_idx, disease_name in enumerate(diseases):
                    for age_idx, age_group_name in enumerate(age_groups):
                        for gender_idx, gender_name in enumerate(genders):
                            for risk_idx, risk_name in enumerate(risks):
                                for run in range(num_runs):
                                    scenario_name = index_dict['scenario_names'][scenario_idx]
                                    year_name = index_dict['time_points'][year_idx]
                                    country_name = countries[country_idx]
                                    scenarios_dalys.append(scenario_name)
                                    years_dalys.append(year_name)
                                    countries_dalys.append(country_name)
                                    diseases_dalys.append(disease_name)
                                    genders_dalys.append(gender_name)
                                    risks_dalys.append(risk_name)
                                    age_groups_dalys.append(age_group_name)
                                    run_values_dalys.append(DALYs_array[scenario_idx, year_idx, country_idx, disease_idx, age_idx, gender_idx, risk_idx, run])

    return pd.DataFrame({
        'Scenario': scenarios_dalys,
        'Year': years_dalys,
        'Country': countries_dalys,
        'Disease': diseases_dalys,
        'Gender': genders_dalys,
        'Risk': risks_dalys,
        'Age Group': age_groups_dalys,
        'DALY Value': run_values_dalys
    })
 