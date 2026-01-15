from helpers import index_dict 

# for uncertainty calculations (for a 1000 runs, num_runs = 1000)
num_runs = 1 # set to 1 if using just the central values 
sample_size = 1000

# setup the file paths the files needed for the calculation
# GBD paths
# input parameters used to construct PAFs 
rf_mord_mort_path = '../Data/GBD 2017/rf_{}_distributions_without_uncertainty.csv' 
TMREL_path = '../Data/GBD 2017/TMREL_distributions_without_uncertainty.csv'
dietary_risk_factors_path = '../Data/GBD 2017/Dietry_risks_relative_risk_factors_parameters.csv'

# input exposure data 
GBD_centralval_path = '../Data/GBD 2017/Central_Values/{}_central_{}.csv'

# shift data containing h values for each scenario, time-point, country, age, sex, and risk 
shift_path = '../Data/Shift/Shift_Example_Emulator.csv' # this is simply a placeholder, users will have to replace this with their own shift file 

# file paths to data used to construct the distributions 
distrbution_weights_path = '../Data/ensemble_distribution_weights.csv'
min_max_path = '../Data/GBD 2017/relative_exposure_minmax.csv'

# input data (total disease burden)
total_YLL_or_YLD_path = '../Data/Projections/total YLLs and YLDs/total_projected_{}s_gendered_new.csv'

# M49 encoding of countries
M49_path = '../Data/Country_Codes_FAO_GBD_ISO_M49.csv'



 