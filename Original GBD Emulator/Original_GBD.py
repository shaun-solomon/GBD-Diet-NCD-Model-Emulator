import numpy as np
import pandas as pd
from helpers_data_and_setup import calculate_mediation_matrix, load_input_files, \
    load_mean_and_std
from helpers_PAF_calculation import full_calculation, calculate_PAF_per_disease
from Distribution_creater_class import DistributionCreator
from Setup_file import num_runs, sample_size, M49_path, index_dict, scenarios, \
    GBD_centralval_path, total_YLL_or_YLD_path, dietary_risk_factors_path, rf_mord_mort_path, TMREL_path 
import time
import sys

'''
################################################
# general setup
################################################
'''

# country M49 code file; used here to derive the list of modelled countries and their UNM49 codes 
country_codes_df = pd.read_csv(M49_path, index_col=3)
country_codes_df.sort_index(inplace=True)

# define model dimensions: countries, risks, diseases, ages, genders 
countries = country_codes_df.loc[country_codes_df['FAO-GBD pair'] == 1].index.values
num_countries = len(countries)

risks = index_dict['risks']
diseases = index_dict['diseases']
num_diseases = len(diseases)

age_groups = index_dict['age_groups']
num_ages = len(age_groups)

genders = index_dict['genders']
num_genders = len(genders)


'''
################################################
# dealing with input arguments from the command line
################################################
'''

# allowing subset execution rather than looping over all countries 
arguments = sys.argv
if len(arguments) != 1:
    start_country_idx = int(arguments[1])
    stop_country_idx = int(arguments[2])
    print(f'It is calculated for countries from {start_country_idx} to {stop_country_idx}')

else:
    print('Only file name was given. The code is executed as normal.')
    start_country_idx = 0
    stop_country_idx = num_countries
    print(start_country_idx, stop_country_idx)
    
'''
################################################
# loading the data
################################################
'''
# get the risk factors dataframe from the csv file
# used to (i) create the risk dictionary, (ii) identify "Both" vs separate morbidity/mortality 
risk_factors_df = pd.read_csv(dietary_risk_factors_path, delimiter=';', index_col=[0, 5, 6])
risk_factors_df.sort_index(inplace=True)

# create risk dictionary, which stores the diseases associated with each risk
risks_dict = {}
for risk in risks:
    risks_dict[risk] = np.unique(risk_factors_df.loc[risk, :].index.get_level_values(0))

# get relative risk (RR) parameter draws for morbidity and mortality 
rf_df_morb = pd.read_csv(rf_mord_mort_path.format('morb'), index_col=[0, 1, 2])
rf_df_morb.sort_index(inplace=True)

rf_df_mort = pd.read_csv(rf_mord_mort_path.format('mort'), index_col=[0, 1, 2])
rf_df_mort.sort_index(inplace=True)

# TMREL values (per risk); columns correspond to draws/runs
TMREL_df = pd.read_csv(TMREL_path, index_col=0)
TMREL_df.sort_index(inplace=True)

# get the mean values of the YLDs and YLLs (those might vary for different scenarios)
total_YLD_df = pd.read_csv(total_YLL_or_YLD_path.format('YLD'), usecols=['location', 'sex', 'cause', 'age', 'val'],
                           index_col=[0, 1, 2, 3])
total_YLD_df.sort_index(inplace=True)

total_YLL_df = pd.read_csv(total_YLL_or_YLD_path.format('YLL'), usecols=['location', 'sex', 'cause', 'age', 'val'],
                           index_col=[0, 1, 2, 3])
total_YLL_df.sort_index(inplace=True)

# create the mediation matrix MF capturing the overlaps between risks (for better overview the calculation happens in helpers_data_and_setup.py)
MF = calculate_mediation_matrix()

# get the bounds and weights used to construct/compose intake distributions 
minmax_bounds_df, distribution_weights_df = load_input_files()

# output arrays
DALYs = np.zeros((num_countries, num_runs))

# stores UNM49 codes corresponding to each country index; used as output index
M49s = np.zeros(num_countries)

for scenario in scenarios:
    print(f"Working on scenario {scenario['name']}")
    
    # loop countries (possibly a subset) for this scenario 
    for idx1, country in enumerate(countries[start_country_idx: stop_country_idx]):
        print(country)
        M49s[idx1] = country_codes_df.loc[country, 'UNM49']
        
        # loading the means and standard deviations (central values only) for this country 
        mean_values_df, sd_values_df = \
            load_mean_and_std(GBD_centralval_path, country)
        begin = time.time()
        
        for run in np.arange(num_runs):
            print(run)
    
            attributable_DALYs = np.zeros((num_diseases, num_ages, num_genders)) # stores DALYs attributable to each outcome/age/sex across risks via joint PAF
            
            # extract exposure means/SDs for this run/draw 
            means = mean_values_df.loc[:, str(run)].to_numpy()
            stds = sd_values_df.loc[:, str(run)].to_numpy()
            
            # generate exposure distributions for each (risk, age, sex) combination for this country and run 
            distributions_df = \
                DistributionCreator(
                    country, risks, age_groups, genders, run, sample_size).get_distributions(
                    means, stds, minmax_bounds_df.loc[country, :], distribution_weights_df)
                        
            # loop through all diseases
            for idx2, disease in enumerate(diseases):
                
                # loop through all age groups
                PAF_array = np.zeros((num_ages, 2, len(risks), 2))
                
                # loop over ages 
                for idx3, age in enumerate(age_groups):
                    
                    # loop over genders 
                    for idx4, gender in enumerate(genders):
                        
                        # pull total burden for this (country,sex, age, disease)
                        total_YLDs = total_YLD_df.loc[(country, gender, age, disease), 'val'] 
                        total_YLLs = total_YLL_df.loc[(country, gender, age, disease), 'val']  
                        
                        # loop over risks 
                        # compute individual PAFs for each risk that applies to this disease
                        for idx5, risk in enumerate(risks):
                            risk_diseases = risks_dict[risk]

                            # only compute if the disease is linked to the current risk 
                            if disease in risk_diseases:
                                morb_mort = \
                                    np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]
                                
                                # if "Both", full_calculation returns both morbidity and mortality PAFs
                                if morb_mort == 'Both':
                                    PAF_value = full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                                 distributions_df, rf_df_morb,
                                                                 rf_df_mort, 'Both', run)
                                    PAF_array[idx3, idx4, idx5, :] = PAF_value
                                    
                                else:
                                    # otherwise compute morbidity and mortality separately 
                                    PAF_array[idx3, idx4, idx5, 0] = \
                                            full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                             distributions_df, rf_df_morb, rf_df_mort, 'Morbidity',run)
                                    PAF_array[idx3, idx4, idx5, 1] = \
                                            full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                             distributions_df, rf_df_morb, rf_df_mort, 'Mortality',run)

                        # aggregate across risks for this disease using the mediation matrix
                        # separate the joint PAFs for morbidity and mortality components 
                        PAF_J_Morb = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 0], MF, idx2)
                        PAF_J_Mort = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 1], MF, idx2)

                        # convert PAFs to attributable DALYs for this disease/age/sex
                        # - morbidity component applied to YLDs
                        # - mortality component applied to YLLs 
                        attributable_DALY = PAF_J_Morb * total_YLDs + PAF_J_Mort * total_YLLs
                        
                        # safety check for invalid values 
                        if np.any(np.isnan(attributable_DALY)):
                            print('value is nan')
                            print(PAF_J_Mort, PAF_J_Morb)
                            exit()
                            
                        attributable_DALYs[idx2, idx3, idx4] = attributable_DALY

            # total attributable DALYs for this country/run (sum over all diseases, ages, sexes)
            DALYs[idx1, run] = np.sum(attributable_DALYs)
            
            # explicit deletes (mostly useful if memory pressure is high)
            del means
            del stds


    time.sleep(1)
    
    # end timing block
    end = time.time()
    print('time', end - begin)

    # create and save dataframe with DALYs per country
    DALYs_df = pd.DataFrame(DALYs, index=M49s, columns=np.arange(num_runs))
    DALYs_df.to_csv(scenario['saving_path'].format(start_country_idx, stop_country_idx), index=True)
         