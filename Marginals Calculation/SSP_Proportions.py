# PROCESSING FILE TO CALCULATE PROPORTIONAL CHANGES IN 
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SSP_NAME = "SSP3"
RAW_SSP_PATH = f"../Data/Demand_SSPs/{SSP_NAME}.txt"
CLEAN_SSP_PATH = f"../Data/Demand_SSPs/Cleaned_Files/{SSP_NAME}_cleaned.txt"
OUTPUT_PROPORTIONS_PATH = f"../Data/Demand_SSPs/SSP_Proportions/{SSP_NAME}_proportions.csv"

TARGET_YEARS = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

# ---------------------------------------------------------------------
# 1. CLEAN & LOAD RAW SSP DEMAND FILE
# ---------------------------------------------------------------------
with open(RAW_SSP_PATH, "r") as f:
    lines = [line.strip().strip('"') for line in f]

with open(CLEAN_SSP_PATH, "w") as f:
    f.write("\n".join(lines))

SSP_demand = pd.read_csv(CLEAN_SSP_PATH, delimiter=",")

# rename columns and filter years
SSP_demand = SSP_demand.rename(columns={"dummy": "year", "dummy.1": "ISO3"})
SSP_demand["year"] = SSP_demand["year"].str.replace("y", "").astype(int)
SSP_demand = SSP_demand[SSP_demand["year"].isin(TARGET_YEARS)]

cols_keep = [
    "year",
    "ISO3",
    "Demand for animal source foods (kcal/capita/day)",
    "Demand for empty calories (kcal/capita/day)",
    "Demand for vegetables fruits and nuts (kcal/capita/day)",
]
SSP_demand = SSP_demand[cols_keep]

# ---------------------------------------------------------------------
# 2. CONVERT TO PROPORTIONS RELATIVE TO 2020
# ---------------------------------------------------------------------
SSP_demand = SSP_demand.set_index(["ISO3", "year"])

baseline = SSP_demand.loc[SSP_demand.index.get_level_values("year") == 2020]
baseline_reindexed = (
    baseline.groupby(level="ISO3")
    .first()
    .loc[SSP_demand.index.get_level_values("ISO3")]
    .set_index(SSP_demand.index)
)

SSP_demand_prop = (SSP_demand / baseline_reindexed).reset_index()

# ---------------------------------------------------------------------
# 3. LOAD INCOME GROUPS AND DEFINE ADJUSTMENTS
# ---------------------------------------------------------------------
income_brackets = pd.read_csv(
    "../Data/Demand_SSPs/Metadata_Country_API_NY.GDP.MKTP.PP.CD_DS2_en_csv_v2_132008.csv",
    usecols=[0, 2],
)
income_brackets = income_brackets.rename(columns={"Country Code": "ISO3"})

# SSP1 adjustment targets 
adjustment_targets = {
    "animal_foods": {
        "Low income": -0.05,
        "Lower middle income": -0.10,
        "Upper middle income": -0.20,
        "High income": -0.30,
    },
    "plant_foods": {
        "Low income": 0.10,
        "Lower middle income": 0.20,
        "Upper middle income": 0.20,
        "High income": 0.30,
    },
    "empty_calories": {
        "Low income": -0.10,
        "Lower middle income": -0.20,
        "Upper middle income": -0.20,
        "High income": -0.30,
    },
    "fatty_acids": {
        "Low income": 0.10,
        "Lower middle income": 0.10,
        "Upper middle income": 0.10,
        "High income": 0.10,
    },
}

"""
# SSP2 adjustment targets 
adjustment_targets = {
    "animal_foods": {
        "Low income": 0.0,
        "Lower middle income": 0.0,
        "Upper middle income": 0.0,
        "High income": 0.0,
    },
    "plant_foods": {
        "Low income": 0.0,
        "Lower middle income": 0.0,
        "Upper middle income": 0.0,
        "High income": 0.0,
    },
    "empty_calories": {
        "Low income": 0.0,
        "Lower middle income": 0.0,
        "Upper middle income": 0.0,
        "High income": 0.0,
    },
    "fatty_acids": {
        "Low income": 0.0,
        "Lower middle income": 0.0,
        "Upper middle income": 0.0,
        "High income": 0.0,
    },
}
"""

# SSP3 adjustment targets 
adjustment_targets = {
    "animal_foods": {
        "Low income": 0.05,
        "Lower middle income": 0.10,
        "Upper middle income": 0.10,
        "High income": 0.05,
    },
    "plant_foods": {
        "Low income": -0.10,
        "Lower middle income": -0.15,
        "Upper middle income": -0.15,
        "High income": -0.10,
    },
    "empty_calories": {
        "Low income": 0.10,
        "Lower middle income": 0.20,
        "Upper middle income": 0.20,
        "High income": 0.10,
    },
    "fatty_acids": {
        "Low income": 0.0,
        "Lower middle income": 0.0,
        "Upper middle income": 0.0,
        "High income": 0.0,
    },
}

"""
# SSP4 adjustment targets 
adjustment_targets = {
    "animal_foods": {
        "Low income": 0.00,
        "Lower middle income": 0.00,
        "Upper middle income": 0.10,
        "High income": 0.05,
    },
    "plant_foods": {
        "Low income": -0.20,
        "Lower middle income": -0.10,
        "Upper middle income": 0.10,
        "High income": 0.20,
    },
    "empty_calories": {
        "Low income": 0.30,
        "Lower middle income": 0.20,
        "Upper middle income": 0.10,
        "High income": -0.10,
    },
    "fatty_acids": {
        "Low income": -0.20,
        "Lower middle income": -0.10,
        "Upper middle income": 0.10,
        "High income": 0.20,
    },
}
"""
"""
# SSP5 adjustment targets 
adjustment_targets = {
    "animal_foods": {
        "Low income": 0.00,
        "Lower middle income": 0.00,
        "Upper middle income": 0.00,
        "High income": 0.00,
    },
    "plant_foods": {
        "Low income": 0.15,
        "Lower middle income": 0.10,
        "Upper middle income": 0.10,
        "High income": 0.10,
    },
    "empty_calories": {
        "Low income": -0.05,
        "Lower middle income": -0.05,
        "Upper middle income": -0.10,
        "High income": -0.10,
    },
    "fatty_acids": {
        "Low income": 0.05,
        "Lower middle income": 0.05,
        "Upper middle income": 0.05,
        "High income": 0.05,
    },
}
"""
merge = pd.merge(SSP_demand_prop, income_brackets, on="ISO3", how="inner")

column_mapping = {
    "Demand for animal source foods (kcal/capita/day)": "animal_foods",
    "Demand for empty calories (kcal/capita/day)": "empty_calories",
    "Demand for vegetables fruits and nuts (kcal/capita/day)": "plant_foods",
}

def compute_adjustment(year, income_group, category_key):
    target = adjustment_targets[category_key].get(income_group, 0)
    years_elapsed = year - 2020
    # Linearly phase in target over 30 years (2020–2050)
    return 1 + (target / 30) * years_elapsed

# apply adjustments for plant/animal/empty
for col, cat_key in column_mapping.items():
    factor_col = f"{col}_adj_factor"
    adj_col = f"{col}_adj"

    merge[factor_col] = merge.apply(
        lambda row: compute_adjustment(row["year"], row["IncomeGroup"], cat_key),
        axis=1,
    )
    merge[adj_col] = merge[col] * merge[factor_col]

# separate fatty-acids factor (no direct column; just an adjustment factor)
merge["fatty_acids_adj_factor"] = merge.apply(
    lambda row: compute_adjustment(row["year"], row["IncomeGroup"], "fatty_acids"),
    axis=1,
)

merge = merge[
    [
        "ISO3",
        "year",
        "IncomeGroup",
        "Demand for animal source foods (kcal/capita/day)_adj",
        "Demand for empty calories (kcal/capita/day)_adj",
        "Demand for vegetables fruits and nuts (kcal/capita/day)_adj",
        "fatty_acids_adj_factor",
    ]
]

merge.to_csv(OUTPUT_PROPORTIONS_PATH, index=False)