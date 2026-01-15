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


def load_std(path, country):
    """
    Loads the standard deviation files (for selected country)
    path (str): relative path where the files are stored
    country (int/str): indicator specifying the country (either name or M49)
    Returns std values dataframe (index: risk factor, age, gender, columns: range(num_runs))
    """
    
    sd_values_df = pd.read_csv(path.format('Std', country), index_col=[0, 1, 2], header=0)
    sd_values_df.sort_index(inplace=True)
    
    return sd_values_df

def load_total_YLDs_YLLs_per_year(path, year): 
    """
    Loads total YLD (Years Lived with Disability) and YLL (Years of Life Lost) dataframes per year from CSV files.
    path (str): The relative path where the files are stored.
    Returns: total_YLD_df, total_YLL_df dataframes indexed by year, location, cause, age, and sex.
    """
    # for YLDs
    total_YLD_df = pd.read_csv(path.format('YLD'), usecols = ['year', 'location', 'sex', 'cause', 'age', 'val'])
    total_YLD_df = total_YLD_df[total_YLD_df['year'] == year]
    total_YLD_df = total_YLD_df.drop(columns='year')
    total_YLD_df = total_YLD_df.set_index(['location', 'sex', 'age', 'cause']).sort_index()
    
    # for YLLs 
    total_YLL_df = pd.read_csv(path.format('YLL'), usecols = ['year', 'location', 'sex', 'cause', 'age', 'val'])
    total_YLL_df = total_YLL_df[total_YLL_df['year'] == year]
    total_YLL_df = total_YLL_df.drop(columns='year')
    total_YLL_df = total_YLL_df.set_index(['location', 'sex', 'age', 'cause']).sort_index()
    
    
    return total_YLD_df, total_YLL_df
    
def load_means_per_year(path, year, country): 
    """
    Loads the means per year for selected country and SSP
    path (str): relative path where the files are stored
    country (int/str): indicator specifying the country (either name or M49)
    Returns mean values dataframe, std values dataframe (index: risk factor, age, gender, columns: range(num_runs))
    """
    mean_df = pd.read_csv(path)
    mean_df.columns = mean_df.columns.map(str)
    
    # Drop saturated fat
    mean_df = mean_df[mean_df["Risk"] != "Diet high in saturated fatty acids"]
    
    mean_df = mean_df[['Risk', 'Year', 'Location', 'Age Group', 'Sex', '0']]
    filtered = mean_df[(mean_df['Location'] == country) & (mean_df['Year'] == year)]
    filtered = filtered.set_index(['Risk', 'Age Group', 'Sex']).sort_index()
    
    return filtered
    

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

 