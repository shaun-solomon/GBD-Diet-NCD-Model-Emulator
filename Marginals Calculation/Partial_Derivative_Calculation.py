import numpy as np   
import pandas as pd
from helpers_data_and_setup import calculate_mediation_matrix, load_input_files, \
    load_std, load_total_YLDs_YLLs_per_year, load_means_per_year
from helpers_PAF_calculation import full_calculation, full_calculation_der, calculate_PAF_der_per_disease
from Distribution_creater_class import DistributionCreator
from Setup_file import num_runs, sample_size, M49_path, index_dict, GBD_centralval_path,\
    total_YLL_or_YLD_path_per_SSP, dietary_risk_factors_path, rf_mord_mort_path, TMREL_path, means_per_SSP
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

# define model dimensions: scenarios, time_points, countries, risks, diseases, ages, genders
time_points = index_dict['time_points']
num_times = len(time_points)

countries = country_codes_df.loc[country_codes_df['FAO-GBD pair'] == 1].index.values
num_countries = len(countries)

risks = index_dict['risks']
num_risks = len(risks)

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

# create the mediation matrix MF capturing the overlaps between risks (for better overview the calculation happens in helpers_data_and_setup.py)
MF = calculate_mediation_matrix()

# get the bounds and weights used to construct/compose intake distributions 
minmax_bounds_df, distribution_weights_df = load_input_files()

# output arrays (one for original marginal DALYs for all ages and one for marginal DALYs below 70)
DALYs_der = np.zeros((num_times, num_countries, num_risks, num_runs))
DALYs_der_below70 = np.zeros_like(DALYs_der)
 
# stores UNM49 codes corresponding to each country index; used as output index
M49s = np.zeros(num_countries)


age_groups_below_70 = [i for i, age in enumerate(age_groups) if int(age.split()[0]) < 70] 

# loop over time-points 
for year_idx, time_point in enumerate(time_points):
    print(f'Calculating for year: {time_point}')
    
    # load the mean values for YLLs and YLDs for that particular time point
    total_YLD_df, total_YLL_df = load_total_YLDs_YLLs_per_year(total_YLL_or_YLD_path_per_SSP, time_point)
    
    # loop over countries
    for idx1, country in enumerate(countries[start_country_idx: stop_country_idx]):
        print(country)
        M49s[idx1] = country_codes_df.loc[country, 'UNM49']
        
        # load the means (for that specific year and SSP) and standard deviations
        mean_values_df = load_means_per_year(means_per_SSP, time_point, country)
        sd_values_df = load_std(GBD_centralval_path, country)
        
        begin = time.time()
        
        # loop over runs
        for run in np.arange(num_runs):
            print(run)
            attributable_DALYs_der = np.zeros((num_diseases, num_ages, num_genders, num_risks))
            
            # extract means and standard deviations for that run 
            means = mean_values_df.loc[:, str(run)].to_numpy()
            stds = sd_values_df.loc[:, str(run)].to_numpy()
            
            # generate exposure distributions for each (risk, age, sex) combination for this country and run 
            distributions_df = \
                DistributionCreator(
                    country, risks, age_groups, genders, run, sample_size).get_distributions(
                    means, stds, minmax_bounds_df.loc[country, :], distribution_weights_df)
                        
            # loop over diseases, age groups, and genders
            for idx2, disease in enumerate(diseases):
                # set up PAF arrays for original and derivatives of PAFs
                PAF_array = np.zeros((num_ages, 2, len(risks), 2))
                PAF_array_der = np.zeros((num_ages, num_genders, num_risks, 2))
                
                
                for idx3, age in enumerate(age_groups):
                    for idx4, gender in enumerate(genders):

                        # loop over risks
                        for idx5, risk in enumerate(risks):
                            risk_diseases = risks_dict[risk]
                            
                            # only compute if the disease is linked to the current risk 
                            if disease in risk_diseases:
                                morb_mort = \
                                    np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]
                                
                                # if "Both", full_calculation and full_calculation_shift returns both
                                if morb_mort == 'Both':
                                    
                                    # compute and store original DALYs
                                    PAF_value = full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                                 distributions_df, rf_df_morb,
                                                                 rf_df_mort, 'Both', run)
                                    PAF_array[idx3, idx4, idx5, :] = PAF_value
                                    
                                    # compute and store PAF derivatives 
                                    PAF_value_der = full_calculation_der(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                                 distributions_df, rf_df_morb,
                                                                 rf_df_mort, 'Both', run)
                                    PAF_array_der[idx3, idx4, idx5, :] = PAF_value_der
                                    
                                else: 
                                    # otherwise compute morbidity and mortality separately
                                    # compute and store original PAFs 
                                    PAF_array[idx3, idx4, idx5, 0] = \
                                        full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                         distributions_df, rf_df_morb, rf_df_mort, 'Morbidity', run)
                                    PAF_array[idx3, idx4, idx5, 1] = \
                                        full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                         distributions_df, rf_df_morb, rf_df_mort, 'Mortality', run)
                                    
                                    # compute and store PAF derivatives     
                                    PAF_array_der[idx3, idx4, idx5, 0] = \
                                        full_calculation_der (risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                         distributions_df, rf_df_morb, rf_df_mort, 'Morbidity', run)
                                    PAF_array_der [idx3, idx4, idx5, 1] = \
                                        full_calculation_der(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                         distributions_df, rf_df_morb, rf_df_mort, 'Mortality', run)
                
                # leave the 'risk' loop after filling the PAF arrays 
                # re-enter the age loop to calculate final PAFs and DALYs 
                for idx3, age in enumerate(age_groups):
                    
                    # loop over genders 
                    for idx4, gender in enumerate(genders):
                        
                        # pull total burden for this (country,sex, age, disease) for that specific time-point
                        total_YLDs = total_YLD_df.loc[(country, gender, age, disease), 'val']  
                        total_YLLs = total_YLL_df.loc[(country, gender, age, disease), 'val']  
                        
                        # loop over risks
                        for idx5, risk in enumerate(risks): 
                            risk_diseases = risks_dict[risk]
                            if disease in risk_diseases:
                                
                                morb_mort = \
                                    np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]

                                # compute PAF derivatives for each risk given the overlap between risks 
                                PAF_J_Morb_der = calculate_PAF_der_per_disease(PAF_array_der[idx3, idx4, :, 0], PAF_array[idx3, idx4, :, 0], MF, idx5, idx2) #
                                PAF_J_Mort_der = calculate_PAF_der_per_disease(PAF_array_der[idx3, idx4, :, 1], PAF_array[idx3, idx4, :, 1], MF, idx5, idx2) 
                            
                            
                             
                                # calculates DALY derivatives 
                                attributable_DALY_der = PAF_J_Morb_der * total_YLDs + PAF_J_Mort_der * total_YLLs
                                if np.any(np.isnan(attributable_DALY_der)):
                                    print('value is nan')
                                    print(PAF_J_Mort_der, PAF_J_Morb_der)
                                    exit()
                                attributable_DALYs_der[idx2, idx3, idx4, idx5] = attributable_DALY_der
                        
                        # loop over risk again
                        for idx5, risk in enumerate(risks): 
                            
                            # save the final DALYs
                            DALYs_der[year_idx, idx1, idx5, run] = np.sum(attributable_DALYs_der[:, :, :, idx5])
                            DALYs_der_below70[year_idx, idx1, idx5, run] = np.sum(attributable_DALYs_der[:, age_groups_below_70, :, idx5]) 

            del means
            del stds

# processing to final dfs
records = []

for year_idx, time_point in enumerate(time_points):
    for idx1, country in enumerate(countries):
        for idx5, risk in enumerate(risks):
            row = {
                'country': country,
                'risk': risk,
                'year': time_point   
            }
            for run in range(num_runs):
                row[run] = DALYs_der[year_idx, idx1, idx5, run]
            records.append(row)

# convert to wide-format DataFrame
DALYs_der_df = pd.DataFrame(records)

records_below70 = []

for year_idx, time_point in enumerate(time_points):
    for idx1, country in enumerate(countries):
        for idx5, risk in enumerate(risks):
            row = {
                'country': country,
                'risk': risk,
                'year': time_point   
            }
            for run in range(num_runs):
                row[run] = DALYs_der_below70[year_idx, idx1, idx5, run]
            records_below70.append(row)

# convert to wide-format DataFrame
DALYs_der_below70_df = pd.DataFrame(records_below70)


DALYs_der_below70_df.columns = DALYs_der_below70_df.columns.map(str)
DALYs_der_df.columns = DALYs_der_df.columns.map(str)

pivoted_df = DALYs_der_df.pivot_table(index=['country', 'risk'], columns='year', values='0').reset_index()
pivoted_df_below_70 = DALYs_der_below70_df.pivot_table(index=['country', 'risk'], columns='year', values='0').reset_index()

# save processed dfs 
pivoted_df.to_csv('../Data/Predictions/Marginals/SSP1/SSP1_all_ages.csv') # change file path per SSP
pivoted_df_below_70.to_csv('../Data/Predictions/Marginals/SSP1/SSP1_below70.csv') # change file path per SSP
