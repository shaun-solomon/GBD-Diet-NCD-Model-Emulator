import numpy as np  
import pandas as pd
from helpers_data_and_setup import calculate_mediation_matrix, calculate_MF_NJ,  calculate_MF_J, load_input_files, \
    load_mean_and_std, convert_to_dataframe
from helpers_PAF_calculation import full_calculation, full_calculation_shift, calculate_PAF_per_disease, \
    change_joint_PAFs_per_disease
from Distribution_creater_class import DistributionCreator
from Setup_file import num_runs, sample_size, M49_path, index_dict, GBD_centralval_path,\
    total_YLL_or_YLD_path, dietary_risk_factors_path, rf_mord_mort_path, shift_path, TMREL_path
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
scenario_names = index_dict['scenario_names']
num_scenarios = len(scenario_names)

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

# Get the shift dataframe containg h values for the different risks 
shift_df = pd.read_csv(shift_path, index_col=[0,1,2,3,4,5])
shift_df.sort_index(inplace=True)
shift_df.index = shift_df.index.set_levels(shift_df.index.levels[1].astype(str), level=1)

# get the mean values of the YLDs and YLLs (those might vary for different scenarios)
total_YLD_df = pd.read_csv(total_YLL_or_YLD_path.format('YLD'), usecols=['year', 'location', 'sex', 'cause', 'age', 'val'],
                           index_col=[0, 1, 2, 3, 4], dtype={'year': str})
total_YLD_df.sort_index(inplace=True)

total_YLL_df = pd.read_csv(total_YLL_or_YLD_path.format('YLL'), usecols=['year', 'location', 'sex', 'cause', 'age', 'val'],
                           index_col=[0, 1, 2, 3, 4], dtype={'year': str})
total_YLL_df.sort_index(inplace=True)

# create the mediation matrix MF capturing the overlaps between risks (for better overview the calculation happens in helpers_data_and_setup.py)
MF = calculate_mediation_matrix()

# get the bounds and weights used to construct/compose intake distributions 
minmax_bounds_df, distribution_weights_df = load_input_files()

# output arrays (one for original GBD DALYs and one for changes in DALYs)
DALYs_per_risk = np.zeros((num_scenarios, num_times, num_countries, num_diseases, num_ages, num_genders, num_risks, num_runs))
DALYs_per_risk_shift = np.zeros((num_scenarios, num_times, num_countries, num_diseases, num_ages, num_genders, num_risks, num_runs))

# stores UNM49 codes corresponding to each country index; used as output index
M49s = np.zeros(num_countries)

# introduces a flag to either execute non-joint or joint PAF calculations (both use the same code by different MFs)
calculate_NJ_DALYs = False

# Loop over scenarios 
for scenario_idx, scenario_name in enumerate(scenario_names):
    # loop over time-points (for projections)
    for year_idx, time_point in enumerate(time_points):
        print(f'Calculating for scenario: {scenario_name}, year: {time_point}')
        
        # loop countries (possibly a subset) for this scenario 
        for idx1, country in enumerate(countries[start_country_idx: stop_country_idx]):
            print(country)
            M49s[idx1] = country_codes_df.loc[country, 'UNM49']
            
            # loading the means and standard deviations (central values only) for this country 
            mean_values_df, sd_values_df = \
                load_mean_and_std(GBD_centralval_path, country)
            
            # Loop over runs (if using central values there would be only one run per scenario, time-point and country)
            for run in np.arange(num_runs):
                print(run)
                
                attributable_DALYs = np.zeros((num_diseases, num_ages, num_genders, num_risks)) # stores DALYs attributable to each outcome/age/sex across risks via joint PAF
                change_attributable_DALYs = np.zeros((num_diseases, num_ages, num_genders, num_risks)) # stores changes in DALYs attributable to each outcome/age/sex across risks via joint PAF
                
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
                    
                    # set up PAF arrays for original and shifted PAFs
                    PAF_array = np.zeros((num_ages, num_genders, num_risks, 2))
                    PAF_array_shift = np.zeros((num_ages, num_genders, num_risks, 2))
                    
                    # loop through age groups 
                    for idx3, age in enumerate(age_groups):
                        
                        # loop through genders 
                        for idx4, gender in enumerate(genders):
                    
                            # loop over risks
                            # compute individual original and shifted PAFs for each risk that applies to this disease 
                            for idx5, risk in enumerate(risks):
                                risk_diseases = risks_dict[risk]
                                
                                # only compute if the disease is linked to the current risk 
                                if disease in risk_diseases:
                                    morb_mort = \
                                        np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]
                                    
                                    # if "Both", full_calculation and full_calculation_shift returns both morbidity and mortality PAFs    
                                    if morb_mort == 'Both':
                                        # compute and store original PAFs
                                        PAF_value = full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                                     distributions_df, rf_df_morb,
                                                                     rf_df_mort, 'Both', run)
                                        PAF_array[idx3, idx4, idx5, :] = PAF_value
                                        # compute and store shifted PAFs 
                                        PAF_value_shift = full_calculation_shift(scenario_name, time_point, country, disease, age, gender, risk, TMREL_df, risk_factors_df,
                                                                         distributions_df, rf_df_morb,
                                                                         rf_df_mort, shift_df, 'Both', run)
                                        PAF_array_shift[idx3, idx4, idx5, :] = PAF_value_shift
                                        
                                    else: 
                                        # otherwise compute morbidity and mortality separately
                                        # compute and store original PAFs 
                                        PAF_array[idx3, idx4, idx5, 0] = \
                                            full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                             distributions_df, rf_df_morb, rf_df_mort, 'Morbidity', run)
                                        PAF_array[idx3, idx4, idx5, 1] = \
                                            full_calculation(risk, disease, age, gender, TMREL_df, risk_factors_df,
                                                             distributions_df, rf_df_morb, rf_df_mort, 'Mortality', run)
                                        
                                        # compute and store shifted PAFs     
                                        PAF_array_shift[idx3, idx4, idx5, 0] = \
                                               full_calculation_shift(scenario_name, time_point, country, disease, age, gender, risk, TMREL_df, risk_factors_df,
                                                                distributions_df, rf_df_morb, rf_df_mort, shift_df, 'Morbidity', run)
                                        PAF_array_shift[idx3, idx4, idx5, 1] = \
                                               full_calculation_shift(scenario_name, time_point,country,disease, age, gender, risk, TMREL_df, risk_factors_df,
                                                                distributions_df, rf_df_morb, rf_df_mort, shift_df, 'Mortality', run)
                    
                    # leave the 'risk' loop after filling the PAF arrays 
                    # re-enter the age loop to calculate final PAFs and DALYs                           
                    for idx3, age in enumerate(age_groups):
                        
                        # loop through genders 
                        for idx4, gender in enumerate(genders):
                                    
                            # pull total burden for this (time point, country,sex, age, disease)
                            total_YLDs = total_YLD_df.loc[(time_point, country, gender, age, disease), 'val']  
                            total_YLLs = total_YLL_df.loc[(time_point, country, gender, age, disease), 'val']  
        
                                    
                            # loop over risks
                            for idx5, risk in enumerate(risks):
                                risk_diseases = risks_dict[risk]
                                
                                if disease in risk_diseases:
                                    morb_mort = \
                                        np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]
                                             
                                    # choose mediation matrix based on the flag (modified mediation matrices are constructed in helpers_data_and_setup.py)
                                    if calculate_NJ_DALYs:
                                        modified_MF = calculate_MF_NJ(risk) # construct matrix for non-joint PAFs
                                        
                                    else: 
                                        modified_MF = calculate_MF_J(MF, risk) # construct matrix for joint PAFs 
                                    
                                    # Calculate individual original PAF for that specific risk 
                                    PAF_J_Morb = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 0], modified_MF, idx2)
                                    PAF_J_Mort = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 1], modified_MF, idx2)
                                    
                                    # Calculate individual shifted PAF for that specific risk 
                                    changes_Morb = change_joint_PAFs_per_disease(PAF_array[idx3, idx4, :, 0], PAF_array_shift[idx3, idx4, idx5, 0], modified_MF, idx2)
                                    changes_Mort = change_joint_PAFs_per_disease(PAF_array[idx3, idx4, :, 1], PAF_array_shift[idx3, idx4, idx5, 1], modified_MF, idx2)

                                    # Calculate attributable DALYs
                                    attributable_DALY = PAF_J_Morb * total_YLDs + PAF_J_Mort * total_YLLs
                                    change_attributable_DALY = changes_Morb * total_YLDs + changes_Mort * total_YLLs
                                    
                                    # Store attributable DALYs
                                    attributable_DALYs[idx2, idx3, idx4, idx5] = attributable_DALY
                                    change_attributable_DALYs[idx2, idx3, idx4, idx5] = change_attributable_DALY
                                    
                # Assign information pertaining to the scenario, year, and country 
                DALYs_per_risk[scenario_idx, year_idx, idx1, :, :, :, :, run] = attributable_DALYs
                DALYs_per_risk_shift[scenario_idx, year_idx, idx1, :, :, :, :, run] = change_attributable_DALYs # relabel
                
                # explicitly delete means and standard deviations (useful if memory pressure is high)
                del means
                del stds

# convert arrays into dataframes 
DALYs_per_risk_df = convert_to_dataframe(DALYs_per_risk, index_dict, diseases, age_groups, genders, risks, countries, num_scenarios, num_times, num_countries, num_diseases, num_ages, num_genders, num_risks, num_runs)
DALYs_per_risk_shift_df = convert_to_dataframe(DALYs_per_risk_shift, index_dict, diseases, age_groups, genders, risks, countries, num_scenarios, num_times, num_countries, num_diseases, num_ages, num_genders, num_risks, num_runs)

# aggregate DALYs to create final dataframe 
DALYs_per_risk_df = DALYs_per_risk_df.groupby(['Scenario', 'Year', 'Country', 'Risk', 'Age Group'])['DALY Value'].sum().reset_index()
DALYs_per_risk_shift_df = DALYs_per_risk_shift_df.groupby(['Scenario', 'Year', 'Country', 'Risk', 'Age Group'])['DALY Value'].sum().reset_index() 

# output processing 
DALYs_per_risk_shift_df.rename(columns = {'DALY Value' : 'DALY Value Change'}, inplace = True) # rename one of the columns 
keys = ['Scenario', 'Year', 'Country', 'Risk']

DALYs_per_risk_df_agg = (DALYs_per_risk_df.groupby(keys, as_index = False)['DALY Value'].sum())
DALYs_per_risk_shift_df_agg = (DALYs_per_risk_shift_df.groupby(keys, as_index = False)['DALY Value Change'].sum())

DALYs_Unilateral_Shift = DALYs_per_risk_df_agg.merge(DALYs_per_risk_shift_df_agg, on = keys, how = 'inner')
DALYs_Unilateral_Shift['DALYs'] = DALYs_Unilateral_Shift['DALY Value'] + DALYs_Unilateral_Shift['DALY Value Change']
DALYs_Unilateral_Shift = DALYs_Unilateral_Shift[['Scenario', 'Year', 'Country', 'Risk', 'DALYs']]
DALYs_Unilateral_Shift.to_csv('../Data/Predictions/Unilateral_Shift/DALYs_J.csv') # change file paths if the flag has been changed 

 

                           
                                            
                
    


