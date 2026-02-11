import pandas as pd
import numpy as np
from Setup_file import risks, diseases, countries, age_groups, genders


def calculate_PAF_per_disease(PAFs, MF, o):
    """
    this function is for calculating the PAFs considering possible overlaps between different risk factors with the
    mediation matrix
    """
    PAF_J = 1 - np.prod(1 - PAFs * np.prod(1 - MF[:, :, o], axis=1))
    return PAF_J

def calculate_mediation_matrix(index_dict):
    """
    this function calculates array mediation factors the risk j mediated by another risk factor i for disease outcome o
    """
    # get mediation matrix file
    med_matrix_df = pd.read_csv('../Data/GBD 2017/mediation_matrix_draw_gbd_2017.csv', index_col=[0, 1, 2])

    # get the IDs of the diseases and risks as DataFrame
    Cause_IDs_df = pd.read_csv('../Data/GBD 2017/IHME_GBD_2019_CAUSE_HIERARCHY_Y2020M11D25.csv',
                               usecols=['Cause_ID', 'Cause_Name'], delimiter=';')
    Cause_IDs_df = Cause_IDs_df.set_index('Cause_Name')

    Risk_IDs_df = pd.read_csv('../Data/GBD 2017/IHME_GBD_2019_REI_HIERARCHY_Y2020M10D15.csv', usecols=['REI_ID', 'REI_Name'],
                              delimiter=';')
    Risk_IDs_df = Risk_IDs_df.set_index('REI_Name')

    diseases = list(index_dict['diseases'])
    other_diseases = np.unique(Cause_IDs_df.index.values)
    diff_diseases_list = list(set(other_diseases).difference(diseases))
    risks = list(index_dict['risks'])
    other_risks = np.unique(Risk_IDs_df.index.values)
    diff_risks_list = list(set(other_risks).difference(risks))

    Cause_IDs_df = Cause_IDs_df.drop(index=diff_diseases_list)
    Risk_IDs_df = Risk_IDs_df.drop(index=diff_risks_list)

    # get IDs for columns to be dropped in the mediation matrix
    Cause_IDs = Cause_IDs_df.loc[:, 'Cause_ID'].tolist()
    other_diseases_IDs = np.unique(med_matrix_df.index.get_level_values(0))
    diff_cause_IDs = list(set(other_diseases_IDs).difference(Cause_IDs))

    Risk_IDs = Risk_IDs_df.loc[:, 'REI_ID'].tolist()
    other_risks_IDs_level1 = np.unique(med_matrix_df.index.get_level_values(1))
    other_risks_IDs_level2 = np.unique(med_matrix_df.index.get_level_values(2))
    diff_risk_IDs_level1 = list(set(other_risks_IDs_level1).difference(Risk_IDs))
    diff_risk_IDs_level2 = list(set(other_risks_IDs_level2).difference(Risk_IDs))

    # swap index column for later use
    Cause_IDs_df.reset_index(inplace=True)
    Risk_IDs_df.reset_index(inplace=True)
    Cause_IDs_df.set_index('Cause_ID', inplace=True)
    Risk_IDs_df.set_index('REI_ID', inplace=True)

    # drop rows which are not necessary in mediation matrix
    med_matrix_df = med_matrix_df.drop(labels=diff_cause_IDs, level='cause_id')
    med_matrix_df = med_matrix_df.drop(labels=diff_risk_IDs_level1, level='rei_id')
    med_matrix_df = med_matrix_df.drop(labels=diff_risk_IDs_level2, level='med_id')

    med_matrix_df['mean'] = np.average(med_matrix_df.loc[:, :'draw_999'].to_numpy(),axis=1)
    med_matrix_df['sd'] = np.std(med_matrix_df.loc[:, :'draw_999'].to_numpy(),axis=1)

    mod_med_matrix_df = med_matrix_df.loc[:, 'med_':]

    # create mediation matrix
    MF = np.zeros((len(risks), len(risks), len(diseases)))

    # get IDs of the diesease, risk and mediation factor triple
    mod_med_matrix_df_idx = med_matrix_df.index.values

    for triple in mod_med_matrix_df_idx:
        idx_dis = diseases.index(Cause_IDs_df.loc[triple[0], 'Cause_Name'])
        idx_risk = risks.index(Risk_IDs_df.loc[triple[1], 'REI_Name'])
        idx_med = risks.index(Risk_IDs_df.loc[triple[2], 'REI_Name'])
        MF[idx_risk, idx_med, idx_dis] = mod_med_matrix_df.loc[triple, 'mean']

    return MF


'''
global settings
'''

with_SBP_med = False;
 
indicators = ['without_additional_salt_factors', 'with_additional_salt_factors'];

for with_salt_factor in range(2):
    
    with_salt_PAF = with_salt_factor;
        
    '''
    append mediation matrix that includes mediation of dietary risk factors by high systolic blood pressure
    '''
    
    index_dict = {'countries': countries, 'risks': risks, 'age_groups': age_groups, 'diseases': diseases,
                  'genders': genders}
    
    if with_SBP_med:
        # to correct to GBD aggregation add high systolic blood pressure (SBP) as risk for mediation
        risks_append = np.append(risks,'High systolic blood pressure');
        # make dictionary to index mediation matrix
        index_dict_append = {'countries': countries, 'risks': risks_append, 'age_groups': age_groups, 'diseases': diseases,
                      'genders': genders}
        # mediation matrix of dietary risks mediated by other dietary risks and SBP 15 x (15+1) matrix
        MF = calculate_mediation_matrix(index_dict_append);
        MF = np.delete(MF,np.size(MF,axis=0)-1,axis=0);
    else:
        MF = calculate_mediation_matrix(index_dict);
    
    '''
    append high in salt disease outcomes mediated by high systolic blood pressure
    '''

    
    risk_factors_df = pd.read_csv('../Data/GBD 2017/Dietry_risks_relative_risk_factors_parameters.csv', delimiter=';',
                                  index_col=[0, 5, 6])
    risk_factors_df.sort_index(inplace=True)
    
    risks_idx_dict = {}
    diseases_idx_dict = {}
    
    for idx, risk in enumerate(risks):
        risks_idx_dict[risk] = idx
    
    for idx, disease in enumerate(diseases):
        diseases_idx_dict[disease] = idx
    
    # load PAFs for additional factors from salt 
    additional_YLDs_df = pd.read_csv('../Data/GBD 2017/additional factors from salt/additional_PAFs_YLDs_salt.csv',
                                     usecols=['location', 'sex', 'cause', 'age', 'val'],
                                     index_col=[0, 1, 2, 3])
    additional_YLDs_df = additional_YLDs_df.swaplevel(1, 3)
    additional_YLDs_df.sort_index(inplace=True)
    additional_YLLs_df = pd.read_csv('../Data/GBD 2017/additional factors from salt/additional_PAFs_YLLs_salt.csv',
                                     usecols=['location', 'sex', 'cause', 'age', 'val'],
                                     index_col=[0, 1, 2, 3])
    additional_YLLs_df = additional_YLLs_df.swaplevel(1, 3)
    additional_YLLs_df.sort_index(inplace=True)
    
    # append diseases to diet high in sodium
    risks_dict = {}
    for risk in risks:
        if risk == 'Diet high in sodium':
            risks_dict[risk] = np.unique(risk_factors_df.loc[risk, :].index.get_level_values(0))
            risks_dict[risk] = np.unique(
                np.append(risks_dict[risk], np.unique(additional_YLDs_df.index.get_level_values(1))))
        else:
            risks_dict[risk] = np.unique(risk_factors_df.loc[risk, :].index.get_level_values(0))
    
    # diet high in sodium attributable fraction of high SBP disease burden
    mediation_dict = {'Intracerebral hemorrhage': 1, 'Ischemic heart disease': 1, 'Ischemic stroke': 1,
                      'Subarachnoid hemorrhage': 1 }
    
    
    # cancel meditation of salt factors with high SBP
    if with_SBP_med:
        for idx, disease in enumerate(diseases):
            print(disease)
            MF[risks_idx_dict['Diet high in sodium'],list(risks_append).index('High systolic blood pressure'), idx] = 0;
    
    
    '''
    calculate the joint dietary risk attributable DALYs with or without correction mediation high systolic blood pressure
    '''
    
    PAF_discrepancies_df = pd.read_csv('../Data/Predictions/Validation/PAF_discrepancies_all_countries.csv',      
                                       index_col=[0, 1, 2, 3], header=[0, 1])
    PAF_discrepancies_df.sort_index(inplace=True)
    
    # get the original PAFs
    original_PAFs_YLDs_df = \
        pd.read_csv('../Data/GBD 2017/YLLs and YLDs attributional to dietary risks/'
                    'attributional_YLDs_gendered_percentage.csv',
                    usecols=['location', 'cause', 'sex', 'age', 'rei', 'val'], index_col=[0, 1, 2, 3, 4])
    original_PAFs_YLDs_df = original_PAFs_YLDs_df.swaplevel(1, 3)
    original_PAFs_YLDs_df.sort_index(inplace=True)
    original_PAFs_YLLs_df = \
        pd.read_csv('../Data/GBD 2017/YLLs and YLDs attributional to dietary risks/'
                    'attributional_YLLs_gendered_percentage.csv',
                    usecols=['location', 'cause', 'sex', 'age', 'rei', 'val'], index_col=[0, 1, 2, 3, 4])
    original_PAFs_YLLs_df = original_PAFs_YLLs_df.swaplevel(1, 3)
    original_PAFs_YLLs_df.sort_index(inplace=True)
    
    # get the population YLLs and YLDs
    total_YLD_YLL_df = pd.read_csv('../Data/Testing/IHME-GBD_2017_DATA-38df4972-1.csv',
                                   usecols=['measure', 'location', 'sex', 'age', 'cause', 'val'],
                                   index_col=[0, 1, 2, 3, 4])
    total_YLD_YLL_df.sort_index(inplace=True)
    
    # get the original DALYs for joint dietary risks
    original_DALYs_df = pd.read_csv(
        '../Data/GBD 2017/YLLs and YLDs attributional to dietary risks/original_attributional_DALYs.csv',
        usecols=['location', 'sex', 'cause', 'age', 'val'],
        index_col=[0, 1, 2, 3])
    original_DALYs_df = original_DALYs_df.swaplevel(1, 3)
    original_DALYs_df.sort_index(inplace=True)
    
    # get the calculated DALYs for join dietary risks
    attributable_DALYs = np.zeros((len(countries), len(diseases), len(age_groups), len(genders)))
    attributable_diffs_DALYs = np.zeros((len(countries), len(diseases), len(age_groups), len(genders)))
    relative_attributable_diffs_DALYs = np.zeros((len(countries), len(diseases), len(age_groups), len(genders)))
    
    # create dataframe for use later and make sure that I do calculate the correct values
    attributable_DALYs_df = pd.DataFrame(index=pd.MultiIndex.from_product([countries, diseases, age_groups, genders]),
                                         columns=['attributable DALYs', 'difference attributable DALYs',
                                                  'relative difference attributable DALYs'])
    attributable_DALYs_df.sort_index(inplace=True)
    
    for idx1, country in enumerate(np.unique(attributable_DALYs_df.index.get_level_values(0))):
        print(country)
    
        for idx2, disease in enumerate(np.unique(attributable_DALYs_df.index.get_level_values(1))):
            PAF_array = np.zeros((len(age_groups), 2, len(risks), 2))
    
            # loop through all age groups
            for idx3, age in enumerate(np.unique(attributable_DALYs_df.index.get_level_values(2))):
                for idx4, gender in enumerate(np.unique(attributable_DALYs_df.index.get_level_values(3))):
                    total_YLDs = \
                        total_YLD_YLL_df.loc[('YLDs (Years Lived with Disability)', country, gender, age, disease),
                                             'val']
                    total_YLLs = \
                        total_YLD_YLL_df.loc[('YLLs (Years of Life Lost)', country, gender, age, disease),
                                             'val']
    
                    for idx5, risk in enumerate(risks):
                        # for each dietary risk get the specific diseases and loop through the diseases
                        risk_diseases = risks_dict[risk]
    
                        if risk == 'Diet high in sodium':
                            if disease == 'Stomach cancer':
                                PAF_array[idx3, idx4, idx5, 0] = \
                                    PAF_discrepancies_df.loc[(country, disease, age, gender),
                                                             ('PAF_discrepancy_j_Morb', risk)] + \
                                    original_PAFs_YLDs_df.loc[(country, disease, age, gender, risk), 'val']
    
                                PAF_array[idx3, idx4, idx5, 1] = \
                                    PAF_discrepancies_df.loc[(country, disease, age, gender),
                                                             ('PAF_discrepancy_j_Mort', risk)] + \
                                    original_PAFs_YLLs_df.loc[(country, disease, age, gender, risk), 'val']
    
                            elif disease in risk_diseases:
                                PAF_array[idx3, idx4, idx5, 0] = \
                                    with_salt_PAF * additional_YLDs_df.loc[(country, disease, age, gender), 'val'] * mediation_dict[disease] 
                                PAF_array[idx3, idx4, idx5, 1] = \
                                    with_salt_PAF * additional_YLLs_df.loc[(country, disease, age, gender), 'val'] * mediation_dict[disease]  
    
                        else:
    
                            # calculate only for the diseases, which actually result from the specific risk factor
                            if disease in risk_diseases:
                                morb_mort = \
                                    np.unique(risk_factors_df.loc[(risk, disease), :].index.get_level_values(0))[0]
    
                                if morb_mort == 'Both':
                                    if (country, disease, age, gender, risk) in original_PAFs_YLDs_df.index.values.tolist():
                                        PAF_array[idx3, idx4, idx5, :] = \
                                            PAF_discrepancies_df.loc[(country, disease, age, gender),
                                                                     ('PAF_discrepancy_j_Morb', risk)] + \
                                            original_PAFs_YLDs_df.loc[(country, disease, age, gender, risk), 'val']
    
                                else:
                                    if (country, disease, age, gender, risk) in original_PAFs_YLDs_df.index.values.tolist():
                                        PAF_array[idx3, idx4, idx5, 0] = \
                                            PAF_discrepancies_df.loc[(country, disease, age, gender),
                                                                     ('PAF_discrepancy_j_Morb', risk)] + \
                                            original_PAFs_YLDs_df.loc[(country, disease, age, gender, risk), 'val']
    
                                    if (country, disease, age, gender, risk) in original_PAFs_YLLs_df.index.values.tolist():
                                        PAF_array[idx3, idx4, idx5, 1] = \
                                            PAF_discrepancies_df.loc[(country, disease, age, gender),
                                                                     ('PAF_discrepancy_j_Mort', risk)] + \
                                            original_PAFs_YLLs_df.loc[(country, disease, age, gender, risk), 'val']
    
                    # calculate the PAFs per disease aggregated over all risks
                    PAF_J_Morb = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 0], MF, idx2)
                    PAF_J_Mort = calculate_PAF_per_disease(PAF_array[idx3, idx4, :, 1], MF, idx2)
    
                    attributable_DALY = PAF_J_Morb * total_YLDs + PAF_J_Mort * total_YLLs
                    original_DALY = original_DALYs_df.loc[(country, disease, age, gender), 'val']
    
                    if np.any(np.isnan(attributable_DALY)):
                        print('value is nan')
                        print(PAF_J_Mort, PAF_J_Morb, total_YLDs, total_YLLs)
                        exit()
                        
                    attributable_DALYs[idx1, idx2, idx3, idx4] = attributable_DALY
                    attributable_diffs_DALYs[idx1, idx2, idx3, idx4] = attributable_DALY - original_DALY
                    relative_attributable_diffs_DALYs[idx1, idx2, idx3, idx4] = \
                        (attributable_DALY - original_DALY) / original_DALY
    
    
    attributable_DALYs_df.loc[:, 'attributable DALYs'] = attributable_DALYs.flatten()
    attributable_DALYs_df.loc[:, 'difference attributable DALYs'] = attributable_diffs_DALYs.flatten()
    attributable_DALYs_df.loc[:, 'relative difference attributable DALYs'] = relative_attributable_diffs_DALYs.flatten()
    
    full_results = attributable_DALYs_df.sort_values(by=['difference attributable DALYs'], ascending=True)
    full_results.to_csv(
        f'../Data/Testing/attributable_DALYs_{indicators[with_salt_factor]}.csv', index=True)
