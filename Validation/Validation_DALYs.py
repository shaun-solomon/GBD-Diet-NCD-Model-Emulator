import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Setup_file import risks, diseases, countries, age_groups
 
def plot_heatmap_of_values(data, idx, idx2, x_entries, x_acronyms, y_entries, y_acronyms, fixed1, fixed2, x_vals,
                           y_vals, relative=False):
    """
    Render and save a heatmap of 95th percentile absolute (or absolute-relative) discrepancies.
    """
    fig, ax = plt.subplots(figsize=(len(x_entries) / 1.5, len(y_entries) / 3))

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_entries)))
    ax.set_xticklabels(x_acronyms)
    ax.set_yticks(np.arange(len(y_entries)))
    ax.set_yticklabels(y_acronyms)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if relative:
        # Loop over data dimensions and create text annotations.
        for i in np.arange(len(y_entries)):
            for j in np.arange(len(x_entries)):
                text = ax.text(j, i, round(data[i, j], 3), ha="center", va="center", color="w", fontsize='xx-small')

        ax.set_title("95th percentile of {}-{} pairs \n - absolute relative difference in DALYs \n {}".format(x_vals, y_vals, idx2))
        fig.tight_layout()
        
        fig.savefig(
            '../Data/Testing/DALY_discrepancy/relative_diff_DALYs_all_{}_all_{}_for_{}_{}_95_percentiles_abs_{}_{}.pdf'.format(
                fixed1, fixed2, x_vals, y_vals, idx, idx2))

    else:
        # Loop over data dimensions and create text annotations.
        for i in np.arange(len(y_entries)):
            for j in np.arange(len(x_entries)):
                text = ax.text(j, i, round(data[i, j], 1), ha="center", va="center", color="w", fontsize='xx-small')

        ax.set_title("95th percentile of {}-{} pairs \n - absolute difference in DALYs \n {}".format(x_vals, y_vals, idx2))
        fig.tight_layout()
        fig.savefig(
            '../Data/Testing/DALY_discrepancy/diff_DALYs_all_{}_all_{}_for_{}_{}_95_percentiles_abs_{}_{}.pdf'.format(
                fixed1, fixed2, x_vals, y_vals, idx, idx2))
        

def calculate_percentiles(df, idx, idx2, x_entries, x_acronyms, y_entries, y_acronyms, x_name, y_name, fixed1, fixed2,
                          relative=False):
    """
    Compute 95th percentile of |discrepancy| for each (y,x) pair and export heatmap + CSV.
    """
    percentile_df = pd.DataFrame(0, index=y_entries, columns=x_entries)
    percentile_df = percentile_df.astype(float);

    for x_val in x_entries:
        for y_val in y_entries:
            discrepancy = df.loc[(y_val), x_val].to_numpy()

            if np.size(discrepancy) != 0:
                percentile = np.percentile(np.abs(discrepancy), 95)
                percentile_df.loc[y_val, x_val] = percentile

    plot_heatmap_of_values(percentile_df.to_numpy(), idx, idx2, x_entries, x_acronyms,
                           y_entries, y_acronyms, fixed1, fixed2, x_name, y_name, relative)

    if relative:
        percentile_df.to_csv(
            '../Data/Testing/DALY_discrepancy/Relative_discrepancy_95_percentiles_abs_{}_{}_pairs_{}_{}.csv'.format(
                x_name, y_name, idx, idx2))
    else:
        percentile_df.to_csv(
            '../Data/Testing/DALY_discrepancy/Discrepancy_95_percentiles_abs_{}_{}_pairs_{}_{}.csv'.format(x_name,
                                                                                                       y_name, idx, idx2))

# setup figure
fig_d, ax_d = plt.subplots(2, 2, figsize=(15, 10));
fig_lim = np.zeros([2,2,2]);
fig_lim[:,:,0]=-np.inf;
fig_lim[:,:,1]=np.inf;
# set limits for the final figure
fig_lim[0,0,0] = -1e6;
fig_lim[0,0,1] = 1e5;

indicators = ['without_additional_salt_factors', 'with_additional_salt_factors']
with_without = ['excluding', 'including']

# load GBD original DALYs
GBD_results_df = pd.read_csv('../Data/GBD 2017/YLLs and YLDs attributional to dietary risks/original_attributional_DALYs.csv',
                                    usecols=['location', 'cause', 'sex', 'age', 'rei', 'val'],
                                     index_col=[0,1,2,3,4])
GBD_results_df = GBD_results_df.swaplevel(1, 3)
GBD_results_df.sort_index(inplace=True)
GBD_results_df = GBD_results_df.stack().unstack(level=0)
original_DALYs = np.sum(GBD_results_df.to_numpy(), axis=0)

for i in range(2):
    # load emulator outputs
    DALYs_diffs_df = pd.read_csv(f'../Data/Testing/attributable_DALYs_{indicators[i]}.csv', index_col=[0, 1, 2, 3])

    cal_DALYs = DALYs_diffs_df.loc[:, ['attributable DALYs']]
    DALY_diffs_df = DALYs_diffs_df.loc[:, ['difference attributable DALYs']]
    relative_DALY_diffs_df = DALYs_diffs_df.loc[:, ['relative difference attributable DALYs']]

    # Create acronyms for labels
    disease_acronyms = np.array([])
    for entry in diseases:
        disease_acronyms = np.append(disease_acronyms, ''.join(w[0] for w in entry.split()))

    age_acronyms = np.array([])
    for entry in age_groups:
        age_acronyms = np.append(age_acronyms, ''.join([entry.split()[0], '-', entry.split()[-1]]))

    '''
    #####################################################
    # plot and save percentile heatmaps per country per disease
    #####################################################
    '''

    # ----------------------------------------------------------------------
    # get distribution for country-disease pairs
    DALY_diffs_df_1 = DALY_diffs_df.stack().unstack(level=1)  # country, age_group , gender, columns = disease
    relative_DALY_diffs_df_1 = relative_DALY_diffs_df.stack().unstack(level=1)  # country, age_group , gender, columns = disease
    
    DALY_diffs_df_1.sort_index(inplace=True)
    relative_DALY_diffs_df_1.sort_index(inplace=True)
    
    for j in range(5):
        sel_countries = countries[j*39:j*39+39]
        calculate_percentiles(DALY_diffs_df_1.loc[sel_countries, :], j, indicators[i],diseases, disease_acronyms, sel_countries, sel_countries,
                              'diseases', 'countries', 'ages', 'gender')
        calculate_percentiles(relative_DALY_diffs_df_1.loc[sel_countries, :], j, indicators[i], diseases, disease_acronyms, sel_countries,
                              sel_countries, 'diseases', 'countries', 'ages', 'gender', True)

    '''
    #####################################################
    # plot and save the distribution of differences across all countries
    #####################################################
    '''

    cal_DALYs = cal_DALYs.stack().unstack(level=0)
    DALYs_diffs_df = DALY_diffs_df.stack().unstack(level=0)  # disease, age_group , gender, columns = country

    total_diff_DALYs = np.sum(DALYs_diffs_df.loc[:, :].to_numpy(), axis=0)
    total_cal_DALYs = np.sum(cal_DALYs.loc[:, :].to_numpy(), axis=0)

    relative_DALYs = total_cal_DALYs/original_DALYs - 1

    total_DALYs_df = pd.DataFrame(np.round(total_diff_DALYs, 1), index=countries, columns=['total diff DALYs'])
    total_DALYs_df.loc[:, 'relative_total_DALYs'] = np.round(relative_DALYs * 100, 2)
    total_DALYs_df.loc[:, 'calculated total DALYs'] = np.round(total_cal_DALYs, 2)

    total_DALYs_df.to_csv(f'../Data/Testing/Diffs_per_country/total_diffs_DALYs_per_country_{indicators[i]}.csv')


    ax_d[0, i].hist(total_diff_DALYs[(total_diff_DALYs > fig_lim[0,i,0]) & (total_diff_DALYs < fig_lim[0,i,1])], alpha=0.5, bins=100, label='DALY difference per country')
    # Hide the right and top spines
    ax_d[0, i].spines.right.set_visible(False)
    ax_d[0, i].spines.top.set_visible(False)
    ax_d[0, i].set_xlabel('differences in DALYs')
    ax_d[0, i].ticklabel_format(axis='x', style='plain')
    ax_d[0, i].legend(frameon=False, loc='best')
    ax_d[0, i].set_title(f'Difference in attributable DALYs per country from \n joint exposure dietary risks ({with_without[i]} sodium factors)')

    ax_d[1, i].hist(relative_DALYs[(total_diff_DALYs > fig_lim[1,i,0]) & (total_diff_DALYs < fig_lim[1,i,1])] * 100, alpha=0.5, bins=100, label='percentage difference per country')
    # Hide the right and top spines
    ax_d[1, i].spines.right.set_visible(False)
    ax_d[1, i].spines.top.set_visible(False)
    ax_d[1, i].set_xlabel('percentage difference in DALYs [%]')
    ax_d[1, i].ticklabel_format(axis='both', style='plain')
    ax_d[1, i].legend(frameon=False, loc='upper left')
    ax_d[1, i].set_title(f'Percentage difference in attributable DALYs per country from \n joint exposure dietary risks ({with_without[i]} sodium factors)')

fig_d.tight_layout()
fig_d.savefig('../Data/Testing/Diffs_per_country/total_diffs_DALYs_per_country_comparison.pdf')
fig_d.clear()
