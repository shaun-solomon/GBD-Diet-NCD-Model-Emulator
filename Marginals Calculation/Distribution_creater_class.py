import numpy as np
from scipy import stats
import sympy
import pandas as pd
from helpers import dist_parameter_tuples, distribution_names
from Variable_creater_class import VariableCreator


def _get_variables(means, stds, min_max_df):
    # computes distribution-specific parameters that cannot be derived directly 
    # from mean and standard deviations alone. 
    variables_df = VariableCreator(means, stds, min_max_df).get_variables_dataframe()
    return variables_df


class DistributionCreator(object):
    #Generates the probability distributions of dietary intake for each (risk,age,gender) combination 
    
    def __init__(self, country, risks, age_groups, genders, run, sample_size=1000):
        # this constructor intitialises the 'DistributionCreator' object 
        # sets up empty dfs for distributions, parameters and coefficients 
        self.country = country
        self.run = run
        self.sample_size = sample_size

        # set up the distributions creator
        # setup the distributions dataframe
        self.distributions_df = pd.DataFrame(index=pd.MultiIndex.from_product([risks, age_groups, genders]),
                                             columns=np.arange(self.sample_size))
        self.distributions_df.sort_index(inplace=True)
        # setup the parameters dataframe
        self.parameters_df = pd.DataFrame(index=self.distributions_df.index,
                                          columns=pd.MultiIndex.from_tuples(dist_parameter_tuples))
        # setup risks, ages, genders
        self.risks = np.unique(self.distributions_df.index.get_level_values(0))
        self.ages = np.unique(self.distributions_df.index.get_level_values(1))
        self.genders = np.unique(self.distributions_df.index.get_level_values(2))
        # setup the coefficients of the single distributions
        self.coefficients = pd.DataFrame(index=risks, columns=distribution_names)

    def get_distributions(self, means, stds, min_max_df, distribution_weights_df):
        self._create_distribution(means, stds, min_max_df, distribution_weights_df)
        return self.distributions_df

    def _create_distribution(self, mean_array, std_array, min_max_df, distribution_weights_df):
        variables_df = _get_variables(mean_array, std_array, min_max_df)
        distributions_array = np.zeros(shape=(len(self.risks) * len(self.ages) * len(self.genders), self.sample_size))
        parameter_array = np.zeros(shape=(len(self.risks) * len(self.ages) * len(self.genders),
                                          len(dist_parameter_tuples)))

        variables_array = variables_df.to_numpy()

        # the index is set to index each risk, age and sex combination
        idx = 0

        for risk in self.risks:
            # get coefficients
            coefficients = distribution_weights_df.loc[risk, 'exp':].to_numpy()
            # if the coefficients do not add up to 1 they get normalised
            coefficients /= coefficients.sum()
            # store coefficients as property of the creator object (depending on risk)
            self.coefficients.loc[risk, :] = coefficients

            for group in self.ages:
                for sex in self.genders:
                    mn = mean_array[idx]
                    sd = std_array[idx]
                    vars = variables_array[idx, :]
                    parameters = self._get_parameters(mn, sd, vars)
                    parameter_array[idx, :] = parameters
                    distributions_array[idx, :] = self._get_distribution(coefficients, parameters)
                    idx += 1

        self.distributions_df.loc[:, :] = distributions_array
        self.parameters_df.loc[:, :] = parameter_array

    def _get_parameters(self, mu, sigma, vars):
        vr = sigma ** 2
        l = vars[9]
        u = vars[8]
        
        parameters = np.zeros(len(dist_parameter_tuples))
        # expon, scale
        parameters[0] = mu
        # gamma, scale & a
        parameters[1] = vr / mu
        parameters[2] = mu ** 2 / vr
        # fisk, scale & c
        parameters[3] = vars[0]
        parameters[4] = vars[1]
        # gumbel_r, scale & loc
        parameters[5] = np.sqrt(6) / np.pi * sigma
        parameters[6] = mu - sigma * np.sqrt(6) / np.pi * float(sympy.EulerGamma.evalf())
        # weibull_min, scale & c
        parameters[7] = vars[3]
        parameters[8] = vars[2]
        # lognorm, scale & s
        parameters[9] = mu ** 2 / np.sqrt(mu ** 2 + vr)
        parameters[10] = np.sqrt(np.log(1 + vr / mu ** 2))
        # norm, scale & loc
        parameters[11] = sigma
        parameters[12] = mu
        # beta, scale, loc, a, & b
        parameters[13] = (vars[10] - vars[11])
        parameters[14] = vars[11]
        parameters[15] = vars[6]
        parameters[16] = vars[7]
        # mirrored_gamma, scale & a
        parameters[17] = vr / (u - mu)
        parameters[18] = (u - mu) ** 2 / vr
        # mirrored_gumbel_r, scale & loc
        parameters[19] = np.sqrt(6) / np.pi * sigma
        parameters[20] = u - mu - sigma * np.sqrt(6) / np.pi * float(sympy.EulerGamma.evalf())
        # invgamma, scale & a
        parameters[21] = mu * (mu**2 / vr + 1)
        parameters[22] = mu**2 / vr + 2
        # invweibull, scale & c
        parameters[23] = 1/vars[5]
        parameters[24] = vars[4]
        # lower l & upper u boundary
        parameters[25] = l
        parameters[26] = u

        return parameters

    def _get_distribution(self, coef, vars):
        u = vars[26]
        l = vars[25]
        np.random.seed(self.run)

        distributions = [
            stats.expon(scale=vars[0]).rvs,  # done
            stats.gamma(scale=vars[1], a=vars[2]).rvs,  # done
            stats.fisk(scale=vars[3], c=vars[4]).rvs,
            stats.gumbel_r(scale=vars[5], loc=vars[6]).rvs,  # done
            stats.weibull_min(scale=vars[7], c=vars[8]).rvs,
            stats.lognorm(scale=vars[9], s=vars[10]).rvs,
            stats.norm(scale=vars[11], loc=vars[12]).rvs,  # done
            stats.beta(scale=vars[13], loc=vars[14], a=vars[15], b=vars[16]).rvs,  # done
            stats.gamma(scale=vars[17], a=vars[18]).rvs,  # done
            stats.gumbel_r(scale=vars[19], loc=vars[20]).rvs,  # done?
            stats.invgamma(scale=vars[21], a=vars[22]).rvs,
            stats.invweibull(scale=vars[23], c=vars[24]).rvs
        ]
        num_distr = len(distributions)
        data = np.zeros((self.sample_size, num_distr))
        
        for idx in range(num_distr):
            if idx == 8 or idx == 9:  # 8 and 9 if beta in
                data[:, idx] = u - distributions[idx](self.sample_size)
            else:
                data[:, idx] = distributions[idx](self.sample_size)
        random_idx = np.random.choice(np.arange(num_distr), size=self.sample_size, p=list(coef))
        samp = data[np.arange(self.sample_size), random_idx]

        while np.any(samp > u) or np.any(samp < l):
            for idx in np.where(samp > u)[0]:
                samp[idx] = data[
                    np.random.choice(np.arange(self.sample_size)), np.random.choice(np.arange(num_distr), p=list(coef))]
            for idx in np.where(samp < l)[0]:
                samp[idx] = data[
                    np.random.choice(np.arange(self.sample_size)), np.random.choice(np.arange(num_distr), p=list(coef))]

        return samp

    def get_pdfs(self, risk, age, gender):
        vars = self.parameters_df.loc[(risk, age, gender), :].to_numpy()
        
         
        distributions = [
            stats.expon(scale=vars[0]).pdf,  # done
            stats.gamma(scale=vars[1], a=vars[2]).pdf,  # done
            stats.fisk(scale=vars[3], c=vars[4]).pdf,
            stats.gumbel_r(scale=vars[5], loc=vars[6]).pdf,  # done
            stats.weibull_min(scale=vars[7], c=vars[8]).pdf,
            stats.lognorm(scale=vars[9], s=vars[10]).pdf,
            stats.norm(scale=vars[11], loc=vars[12]).pdf,  # done
            stats.beta(scale=vars[13], loc=vars[14], a=vars[15], b=vars[16]).pdf,  # done
            stats.gamma(scale=vars[17], a=vars[18]).pdf,  # done
            stats.gumbel_r(scale=vars[19], loc=vars[20]).pdf,  # done?
            stats.invgamma(scale=vars[21], a=vars[22]).pdf,
            stats.invweibull(scale=vars[23], c=vars[24]).pdf
        ]
        
        return distributions

    # this function is not called, but can be useful to assess the rvs for testing
    def get_rvss(self, risk, age, gender):
        vars = self.parameters_df.loc[(risk, age, gender), :].to_numpy()

        distributions = [
            stats.expon(scale=vars[0]).rvs,  # done
            stats.gamma(scale=vars[1], a=vars[2]).rvs,  # done
            stats.fisk(scale=vars[3], c=vars[4]).rvs,
            stats.gumbel_r(scale=vars[5], loc=vars[6]).rvs,  # done
            stats.weibull_min(scale=vars[7], c=vars[8]).rvs,
            stats.lognorm(scale=vars[9], s=vars[10]).rvs,
            stats.norm(scale=vars[11], loc=vars[12]).rvs,  # done
            stats.beta(scale=vars[13], loc=vars[14], a=vars[15], b=vars[16]).rvs,  # done
            stats.gamma(scale=vars[17], a=vars[18]).rvs,  # done
            stats.gumbel_r(scale=vars[19], loc=vars[20]).rvs,  # done?
            stats.invgamma(scale=vars[21], a=vars[22]).rvs,
            stats.invweibull(scale=vars[23], c=vars[24]).rvs
        ]

        return distributions

