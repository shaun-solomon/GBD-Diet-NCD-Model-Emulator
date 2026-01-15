import numpy as np
import pandas as pd
from helpers_variables_calculation import run_for_beta, run_for_fisk, run_for_invweibull, run_for_weibull, \
    greater_one_weibull, smaller_one_eighth_weibull, greater_4_5_inv_weibull, smaller_one_eighth_invweibull
from helpers import index_dict


class VariableCreator(object):
    def __init__(self, means, stds, min_max_df):
        """
        means_df (dataframe): contains the means for different risk, age and sex combinations for specific country and run
                              index = Multiindex (risks, ages, sexes),
                              columns = only one, the run has already been selected before
                              => run number and country have to be selected before
        stds_df (dataframe): contains the stds for different risk, age and sex combinations for specific country and run
                             index = Multiindex (risks, ages, sexes),
                             columns = only one, the run has already been selected before
                             => run number and country have to be selected before
        min_max_df (dataframe): contains the min and max values for different risk, age and sex combinations
                                for specific country and run
                                index = Multiindex (risks, ages, sexes),
                                columns = only one, the run has already been selected before
                                => run number and country have to be selected before
        """
        risks = index_dict['risks']
        ages = index_dict['age_groups']
        genders = index_dict['genders']

        # setup the variables dataframe
        self.variables_df = pd.DataFrame(index=pd.MultiIndex.from_product([risks, ages, genders]),
                                         columns=['alpha_fisk', 'beta_fisk', 'k_weibull', 'lambda_weibull',
                                                  'k_invweibull', 'lambda_invweibull', 'a_beta', 'b_beta', 'u', 'l',
                                                  'u_beta', 'l_beta'])
        self.variables_df.sort_index(inplace=True)
        self.risks = np.unique(self.variables_df.index.get_level_values(0))
        self.ages = np.unique(self.variables_df.index.get_level_values(1))
        self.genders = np.unique(self.variables_df.index.get_level_values(2))

        # fill the variables dataframe
        self._calculate_variables_beta(means, stds, min_max_df=min_max_df)
        self._calculate_variables_with_optimisation(means, stds)

    def _calculate_variables_beta(self, means, stds, min_max_df):
        """
        means_df (dataframe): contains the means for different risk, age and sex combinations for specific country and run
                              index = Multiindex (risks, ages, sexes),
                              columns = only one, the run has already been selected before
                              => run number and country have to be selected before
        stds_df (dataframe): contains the stds for different risk, age and sex combinations for specific country and run
                             index = Multiindex (risks, ages, sexes),
                             columns = only one, the run has already been selected before
                             => run number and country have to be selected before
        """
        mean = means.flatten()
        std = stds.flatten()

        upper = min_max_df.loc[:, 'xmax'].to_numpy() * mean
        lower = min_max_df.loc[:, 'xmin'].to_numpy() * mean

        self.variables_df.loc[:, 'u'] = upper
        self.variables_df.loc[:, 'l'] = lower

        a_beta, b_beta, u_beta, l_beta = run_for_beta(mean, std, upper, lower)
        if np.any(np.isnan(a_beta)):
            print('problem')
        self.variables_df.loc[:, 'a_beta'] = a_beta.T
        self.variables_df.loc[:, 'b_beta'] = b_beta.T
        self.variables_df.loc[:, 'u_beta'] = u_beta.T
        self.variables_df.loc[:, 'l_beta'] = l_beta.T

    def _calculate_variables_with_optimisation(self, means, stds):
        '''
        means_df (dataframe): contains the means for different risk, age and sex combinations for specific country and run
                              index = Multiindex (risks, ages, sexes),
                              columns = only one, the run has already been selected before
                              => run number and country have to be selected before
        stds_df (dataframe): contains the stds for different risk, age and sex combinations for specific country and run
                             index = Multiindex (risks, ages, sexes),
                             columns = only one, the run has already been selected before
                             => run number and country have to be selected before
        '''
        variables_array = np.zeros_like(self.variables_df.loc[:, 'alpha_fisk':'lambda_invweibull'].to_numpy())

        # select values for optimisation in weibull
        selection_i = np.where(means/stds >= 1)[0]
        selection_inv_i = np.where(means/stds >= 4.5)[0]
        selection_ii = np.where(means/stds <= 1/8)[0]

        k, lamb = greater_one_weibull(means[selection_i], stds[selection_i])
        k_inv, lamb_inv = greater_4_5_inv_weibull(means[selection_inv_i], stds[selection_inv_i])
        variables_array[selection_i, 2] = k.T
        variables_array[selection_i, 3] = lamb.T
        variables_array[selection_inv_i, 4] = k_inv.T
        variables_array[selection_inv_i, 5] = lamb_inv.T

        k, lamb = smaller_one_eighth_weibull(means[selection_ii], stds[selection_ii])
        k_inv, lamb_inv = smaller_one_eighth_invweibull(means[selection_ii], stds[selection_ii])
        variables_array[selection_ii, 2] = k.T
        variables_array[selection_ii, 3] = lamb.T
        variables_array[selection_ii, 4] = k_inv.T
        variables_array[selection_ii, 5] = lamb_inv.T

        selection_iii = \
            [index for index in range(len(means)) if index not in np.append(selection_i, selection_ii)]
        selection_inv_iii = \
            [index for index in range(len(means)) if index not in np.append(selection_inv_i, selection_ii)]

        # for fisk
        for idx in range(len(np.unique(self.variables_df.loc[:, 'alpha_fisk':'lambda_invweibull'].index.values))):
            mean = means[idx]
            sd = stds[idx]
            variables_array[idx, :2] = run_for_fisk(mean, sd)
            if idx in selection_iii:
                variables_array[idx, 2:4] = run_for_weibull(mean, sd)
            if idx in selection_inv_iii:
                variables_array[idx, 4:6] = run_for_invweibull(mean, sd)

        self.variables_df.loc[:, 'alpha_fisk':'lambda_invweibull'] = variables_array

    def get_variables_dataframe(self):
        return self.variables_df

