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
