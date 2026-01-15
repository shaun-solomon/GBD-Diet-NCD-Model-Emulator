from helpers import index_dict 

# for uncertainty calculations (for a 1000 runs, num_runs = 1000)
num_runs = 1 #set to 1 if using central values 
sample_size = 1000

# setup the file paths the files needed for the calculation
# GBD paths
# input parameters used to construct PAFs 
rf_mord_mort_path = '../Data/GBD 2017/rf_{}_distributions_without_uncertainty.csv'
TMREL_path = '../Data/GBD 2017/TMREL_distributions_without_uncertainty.csv'
dietary_risk_factors_path = '../Data/GBD 2017/Dietry_risks_relative_risk_factors_parameters.csv'

# input exposure data 
GBD_centralval_path = '../Data/GBD 2017/Central_Values/{}_central_{}.csv'
means_per_SSP = '../Data/SSP Means/SSP_means/SSP1_means.csv' # needs to be changed for every SSP
 
# file paths to data used to construct the distributions 
distrbution_weights_path = '../Data/ensemble_distribution_weights.csv'
min_max_path = '../Data/GBD 2017/relative_exposure_minmax.csv'

# input data (total disease burden)
# needs to be changed for every SSP 
total_YLL_or_YLD_path_per_SSP = '../Data/SSP_YLL_YLD_Projections/SSPs/SSP1/SSP1_total_{}s_projected_gendered.csv'  
 
# M49 encoding of countries
M49_path = '../Data/Country_Codes_FAO_GBD_ISO_M49.csv'



 
 