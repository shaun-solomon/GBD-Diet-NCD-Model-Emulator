import pandas as pd
import argparse
import os


def build_gendered_yld_projections():
    """
    Build sex-specific YLD projections (central values) from GBD projections and 2021 splits,
    with uncertainty bounds propagated from 2021 ratios.
    Returns a DataFrame with columns:
    ['year', 'location', 'cause', 'age', 'sex', 'val', 'lower', 'upper']
    for years 2025–2050.
    """
    # load GBD YLD projections ----------
    projections = pd.read_csv('../Data/SSP_YLL_YLD_Projections/YLD.csv') # replace with YLL file if needed 

    # select rows where age is 25+ years
    selected_age_id = [10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 30, 31, 32, 235]
    projections = projections[projections['age_group_id'].isin(selected_age_id)]

    # load meta data
    age_meta = pd.read_csv('../Data/SSP_YLL_YLD_Projections/meta_data/meta_age_group.csv')
    cause_meta = pd.read_csv('../Data/SSP_YLL_YLD_Projections/meta_data/meta_cause.csv')
    locations_meta = pd.read_csv('../Data/SSP_YLL_YLD_Projections/meta_data/meta_locations.csv')
    sex_meta = pd.read_csv('../Data/SSP_YLL_YLD_Projections/meta_data/meta_sex.csv')

    # merge projections with metadata
    projections_fin = pd.merge(projections, age_meta, on='age_group_id', how='left')
    projections_fin = pd.merge(projections_fin, locations_meta, on='location_id', how='left')
    projections_fin = pd.merge(projections_fin, sex_meta, on='sex_id', how='left')
    projections_fin = pd.merge(projections_fin, cause_meta, on='acause', how='left')

    # drop and tidy columns
    projections_fin.drop(
        columns=['statistic', 'scenario', 'sex_id', 'age_group_id',
                 'location_id', 'cause_id', 'acause'],
        inplace=True
    )
    projections_fin.rename(
        columns={
            'year_id': 'year',
            'age_group_name': 'age',
            'location_name': 'location',
            'cause_name': 'cause'
        },
        inplace=True
    )
    projections_fin = projections_fin[['location', 'age', 'cause', 'sex', 'value', 'year']]

    # load 2021 YLD values and compute sex proportions 
    YLD_2021 = pd.read_csv('../Data/GBD 2021/total YLLs and YLDs/total_YLDs_gendered.csv')

    # group by location, age and cause, summing over sex
    grouped = YLD_2021.groupby(['location', 'age', 'cause'])
    YLD_2021['total_val'] = grouped['val'].transform('sum')
    YLD_2021['total_val_upper'] = grouped['upper'].transform('sum')
    YLD_2021['total_val_lower'] = grouped['lower'].transform('sum')

    # proportions per sex
    YLD_2021['proportions'] = YLD_2021['val'] / YLD_2021['total_val']
    YLD_2021['proportions_upper'] = YLD_2021['upper'] / YLD_2021['total_val_upper']
    YLD_2021['proportions_lower'] = YLD_2021['lower'] / YLD_2021['total_val_lower']

    YLD_2021.drop(
        columns=['total_val', 'total_val_upper', 'total_val_lower',
                 'measure', 'metric', 'year', 'val', 'upper', 'lower',
                 'proportions_upper', 'proportions_lower'],
        inplace=True
    )
    YLD_2021 = YLD_2021[['location', 'age', 'cause', 'sex', 'proportions']]

    # normalise age labels to match metadata if needed
    YLD_2021['age'] = (
        YLD_2021['age']
        .str.replace(' years', '', regex=False)
        .str.replace('-', ' to ', regex=False)
        .str.replace('+', ' plus', regex=False)
    )


    # expand "Both" projections into Male/Female using 2021 proportions ----------
    projections_expanded = pd.concat(
        [
            projections_fin[projections_fin['sex'] == 'Both'].assign(sex=sex)
            for sex in ['Male', 'Female']
        ],
        ignore_index=True
    )

    merged = pd.merge(
        projections_expanded,
        YLD_2021,
        on=['location', 'age', 'cause', 'sex'],
        how='left'
    )

    # central value split by sex
    merged['value'] = merged['value'] * merged['proportions']
    merged.drop(columns=['proportions'], inplace=True)

    merged = merged[['location', 'age', 'cause', 'sex', 'value', 'year']]
    merged = merged.sort_values(by=['location', 'age', 'cause', 'sex', 'year'])

    # keep projection years only
    projections_gendered = merged[merged['year'].isin([2025, 2030, 2035, 2040, 2045, 2050])]

    # add uncertainty bounds via 2021 ratios ----------
    YLD_2021_bounds = pd.read_csv('../Data/GBD 2021/total YLLs and YLDs/total_YLDs_gendered.csv')

    YLD_2021_bounds['age'] = (
        YLD_2021_bounds['age']
        .str.replace(' years', '', regex=False)
        .str.replace('-', ' to ', regex=False)
        .str.replace('+', ' plus', regex=False)
    )

    YLD_2021_bounds['lower_ratio'] = YLD_2021_bounds['lower'] / YLD_2021_bounds['val']
    YLD_2021_bounds['upper_ratio'] = YLD_2021_bounds['upper'] / YLD_2021_bounds['val']

    ratios_df = YLD_2021_bounds[['location', 'age', 'sex', 'cause', 'lower_ratio', 'upper_ratio']]

    merged_bounds = pd.merge(
        projections_gendered,
        ratios_df,
        on=['location', 'age', 'sex', 'cause'],
        how='inner'
    )

    merged_bounds['lower'] = merged_bounds['value'] * merged_bounds['lower_ratio']
    merged_bounds['upper'] = merged_bounds['value'] * merged_bounds['upper_ratio']

    merged_bounds = merged_bounds[['year', 'location', 'cause', 'age', 'sex', 'value', 'lower', 'upper']]
    merged_bounds.rename(columns={'value': 'val'}, inplace=True)

    return merged_bounds


def add_baseline_2020(merged_bounds: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2020 baseline (from 2021 GBD totals) to the projected YLDs.
    Returns stacked DataFrame with years 2020 + 2025–2050.
    """
    baseline = pd.read_csv('../Data/GBD 2021/total YLLs and YLDs/total_YLDs_gendered.csv')

    baseline['age'] = (
        baseline['age']
        .str.replace(' years', '', regex=False)
        .str.replace('-', ' to ', regex=False)
        .str.replace('+', ' plus', regex=False)
    )

    baseline.drop(columns=['measure', 'metric', 'year'], inplace=True)
    baseline['year'] = 2020

    # align column order to merged_bounds
    baseline = baseline[['year', 'location', 'cause', 'age', 'sex', 'val', 'lower', 'upper']]

    stacked_df = pd.concat([merged_bounds, baseline], ignore_index=True)
    stacked_df = stacked_df.sort_values(by=['location', 'cause', 'age', 'sex', 'year'])

    return stacked_df


def apply_ssp_scaling(stacked_df: pd.DataFrame, ssp: str) -> pd.DataFrame:
    """
    Apply SSP-specific population proportions to YLDs.
    ssp: one of 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'
    Returns DataFrame with SSP-scaled val_SSP, lower_SSP, upper_SSP.
    """
    if ssp not in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        raise ValueError("ssp must be one of: SSP1, SSP2, SSP3, SSP4, SSP5")

    SSP_pop = pd.read_csv(
        '../Data/f09_pop_iso(in).csv',
        skiprows=4,
        usecols=['year', 'ISO3', 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    )

    # SSP proportions relative to SSP2
    SSP_pop['SSP1_prop'] = SSP_pop['SSP1'] / SSP_pop['SSP2']
    SSP_pop['SSP5_prop'] = SSP_pop['SSP5'] / SSP_pop['SSP2']
    SSP_pop['SSP3_prop'] = SSP_pop['SSP3'] / SSP_pop['SSP2']
    SSP_pop['SSP4_prop'] = SSP_pop['SSP4'] / SSP_pop['SSP2']
    SSP_pop['SSP2_prop'] = SSP_pop['SSP2'] / SSP_pop['SSP2']

    # Convert 'y1965' → integer
    SSP_pop['year'] = SSP_pop['year'].str.replace('y', '').astype(int)

    # Filter 2020–2050
    SSP_pop = SSP_pop[SSP_pop['year'].between(2020, 2050)]

    # Attach GBD locations
    Country_Codes = pd.read_csv(
        '../Data/Country_Codes_FAO_GBD_ISO_M49.csv',
        usecols=['GBD_name', 'ISO3']
    )
    Country_Codes.rename(columns={'GBD_name': 'location'}, inplace=True)

    merged_pop = pd.merge(SSP_pop, Country_Codes, on='ISO3', how='left')

    prop_col = f'{ssp}_prop'
    if prop_col not in merged_pop.columns:
        raise KeyError(f"Column {prop_col} not found in SSP population file.")

    ssp_df = merged_pop[['year', 'location', prop_col]]

    # merge with YLD stacked data
    final = pd.merge(
        stacked_df,
        ssp_df,
        on=['year', 'location'],
        how='inner'
    )

    final['val_SSP'] = final['val'] * final[prop_col]
    final['lower_SSP'] = final['lower'] * final[prop_col]
    final['upper_SSP'] = final['upper'] * final[prop_col]
    
    final = final[['year', 'location', 'cause', 'age', 'sex', 'val', 'lower', 'upper']]
    
    return final


def main(ssp: str):
    # build gendered YLD projections (central + bounds) for projection years
    merged_bounds = build_gendered_yld_projections()

    # add 2020 baseline
    stacked_df = add_baseline_2020(merged_bounds)

    # apply SSP scaling
    final_ssp_df = apply_ssp_scaling(stacked_df, ssp)

    # save to SSP-specific output
    out_dir = f'../Data/SSP_YLL_YLD_Projections/SSPs/{ssp}'
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'{ssp}_total_YLDs_projected_gendered.csv') # change out path for YLLs
    final_ssp_df.to_csv(out_path, index=False)
    print(f"Saved SSP-scaled YLD projections to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute SSP-specific projected YLDs (gendered, with bounds) from GBD projections."
    )
    parser.add_argument(
        '--ssp',
        type=str,
        required=True,
        help="SSP scenario name (one of: SSP1, SSP2, SSP3, SSP4, SSP5)"
    )
    args = parser.parse_args()
    main(args.ssp)