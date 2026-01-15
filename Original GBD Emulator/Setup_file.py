from helpers import index_dict 

# for uncertainty calculations (for a 1000 runs, num_runs = 1000)
num_runs = 1 # set to 1 if using just the central values 
sample_size = 1000

# set unit for marginal
unit_of_marginal = 'DALYs'

# setup the file paths the files needed for the calculation
# GBD paths
# input parameters used to construct PAFs 
rf_mord_mort_path = '../Data/GBD 2017/rf_{}_distributions_without_uncertainty.csv' 
TMREL_path = '../Data/GBD 2017/TMREL_distributions_without_uncertainty.csv'
dietary_risk_factors_path = '../Data/GBD 2017/Dietry_risks_relative_risk_factors_parameters.csv'

# input exposure data 
GBD_centralval_path = '../Data/GBD 2017/Central_Values/{}_central_{}.csv'

# file paths to data used to construct the distributions 
distrbution_weights_path = '../Data/ensemble_distribution_weights.csv'
min_max_path = '../Data/GBD 2017/relative_exposure_minmax.csv'

# input data (total disease burden)
total_YLL_or_YLD_path = '../Data/GBD 2017/total YLLs and YLDs/total_{}s_gendered.csv'

# M49 encoding of countries
M49_path = '../Data/Country_Codes_FAO_GBD_ISO_M49.csv'

# path were the predictions are saved)
directory_path = '../Data/Predictions/Original_GBD/'

# create list of scenario dicts
scenarios = [{'name': 'original_GBD',
              'indicator': '',
              'path': '../../Regression Model/Final_model/Final_predictions/Original_GBD/Per_country'
                      '/Predictions_GBD_{}_{}.csv',
              'description': 'Full GBD data for a specific risk for year 2019 (needs to be formatted)',
              'directory_path': directory_path,
              'saving_path': directory_path + 'DALY_predictions_{}_{}.csv',
              'full_saving_path': directory_path + 'Full_DALY_predictions.csv',
              # those two are for testing reasons only should be filled with 0s!
              'diffs_saving_path': directory_path + 'Full_marginal_DALY_changes.csv',
              'costs_saving_path': directory_path + 'Full_marginal_costs.csv'}]

 