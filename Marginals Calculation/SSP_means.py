# script generates SSP specific means used as inputs for the final code 
import pandas as pd

SSP_NAME = "SSP1"
PROPORTIONS_PATH = f"../Data/SSP Means/SSP_Proportions/{SSP_NAME}_proportions.csv"
OUTPUT_MEANS_PATH = f"../Data/SSP Means/SSP_means/{SSP_NAME}_means.csv"

# ---------------------------------------------------------------------
# 1. LOAD SSP PROPORTIONS
# ---------------------------------------------------------------------
SSP_proportions = pd.read_csv(PROPORTIONS_PATH)

rename_columns = {
    "Demand for animal source foods (kcal/capita/day)_adj": "animal",
    "Demand for empty calories (kcal/capita/day)_adj": "empty",
    "Demand for vegetables fruits and nuts (kcal/capita/day)_adj": "plant",
    "fatty_acids_adj_factor": "fatty_acids",
}
SSP_proportions = SSP_proportions.rename(columns=rename_columns)

SSP_proportions_long = SSP_proportions.melt(
    id_vars=["ISO3", "year", "IncomeGroup"],
    var_name="category",
    value_name="proportion",
)
SSP_proportions_long = SSP_proportions_long.rename(columns={"year": "Year"})

# ---------------------------------------------------------------------
# 2. LOAD BASELINE MEAN INTAKES (2017) AND EXPAND OVER YEARS
# ---------------------------------------------------------------------
mean_intakes = pd.read_csv("../Data/GBD 2017/Central_Values/dietary_means_2017_corrected.csv")
mean_intakes = mean_intakes.drop(columns="Year")

years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
years_df = pd.DataFrame({"Year": years})
years_df["key"] = 1

mean_intakes["key"] = 1
mean_intakes_expanded = mean_intakes.merge(years_df, on="key").drop(columns="key")

# ---------------------------------------------------------------------
# 3. MAP LOCATIONS TO ISO3
# ---------------------------------------------------------------------
Country_Codes = pd.read_csv(
    "../Data/Country_Codes_FAO_GBD_ISO_M49.csv", usecols=["ISO3", "GBD_2017_name"]
)
Country_Codes = Country_Codes.rename(columns={"GBD_2017_name": "Location"})

mean_intakes_expanded_ISO3 = pd.merge(
    mean_intakes_expanded, Country_Codes, on="Location", how="inner"
)

# ---------------------------------------------------------------------
# 4. MAP RISKS TO DEMAND CATEGORIES
# ---------------------------------------------------------------------
risk_category_map = {
    "Diet high in red meat": "animal",
    "Diet high in processed meat": "empty",
    "Diet low in milk": "animal",
    "Diet low in calcium": "animal",
    "Diet low in fiber": "plant",
    "Diet low in seafood omega-3 fatty acids": "fatty_acids",
    "Diet low in fruits": "plant",
    "Diet low in whole grains": "plant",
    "Diet low in legumes": "plant",
    "Diet low in nuts and seeds": "plant",
    "Diet low in polyunsaturated fatty acids": "fatty_acids",
    "Diet high in sodium": "empty",
    "Diet high in sugar-sweetened beverages": "empty",
    "Diet high in trans fatty acids": "empty",
    "Diet low in vegetables": "plant",
}

mean_intakes_expanded_ISO3["category"] = mean_intakes_expanded_ISO3["Risk"].map(
    risk_category_map
)

# ---------------------------------------------------------------------
# 5. MERGE PROPORTIONS AND CALCULATE PROJECTED MEANS
# ---------------------------------------------------------------------
projected_df = mean_intakes_expanded_ISO3.merge(
    SSP_proportions_long, on=["ISO3", "Year", "category"], how="left"
)

projected_df["Mean_projected"] = projected_df["Mean"] * projected_df["proportion"]
projected_df["Lower_projected"] = projected_df["Lower"] * projected_df["proportion"]
projected_df["Upper_projected"] = projected_df["Upper"] * projected_df["proportion"]

projected_df = projected_df[
    [
        "Risk",
        "Year",
        "Location",
        "Age Group",
        "Sex",
        "Unit",
        "Mean_projected",
        "Lower_projected",
        "Upper_projected",
    ]
]

projected_df.to_csv(OUTPUT_MEANS_PATH, index=False)

