# GBD Diet-NCD Emulator (2017)
## Overview 

The **GBD Diet–NCD Model** (developed by IHME) quantifies the impact of suboptimal diets on non communicable disease (NCD) morbidity and mortality (1). This repository provides a Python implementation that **reproduces the core results of GBD 2017** and extends them to **scenario-based and marginal (partial derivative)** analyses. The emulator constructs **Population Attributable Fractions (PAFs)** for each dietary risk using (i) exposure distributions (means & standard deviations), (ii) **relative risks (RRs)** and **TMRELs** (Theoretical Minimum Risk Exposure Levels), and (iii) **disease burden inputs** (YLLs, YLDs) to derive **DALYs**.

Although IHME has since released newer GBD vintages, the 2017 inputs allow a **self-contained, transparent** workflow. The codebase implements deterministic calculations (central values) with consistent indexing across **country, sex, age group, risk, disease outcome**, and optionally, **time/scenario** for projections.

### Table 1 - Dietary risk factors included in GBD 2017 and direction of risk (2) 
| Dietary Risk Factors | Direction of Risk |
| ---------------------| ----------------- |
| Diet low in fruits | Low intake|
| Diet low in vegetables | Low intake|
| Diet low in legumes | Low intake | 
| Diet low in whole grains | Low intake | 
| Diet low in nuts and seeds | Low intake|
| Diet low in milk | Low intake |
| Diet high in red meat | High intake |
| Diet high in processed meat | High intake | 
| Diet high in sugar-sweetened beverages | High intake | 
| Diet low in fiber | Low intake |
| Diet low in calcium | Low intake |
| Diet low in seafood omega-3 fatty acids | Low intake |
| Diet low in polyunsaturated fatty acids | Low intake |
| Diet high in trans fatty acids | High intake |
| Diet high in sodium | High intake | 

## What the Emulator Can Do

The emulator can implement the following analytical scenarios: 

### Analytical Scenario 1: 

Reproduces disease burden **(YLLs, YLDs, and DALYs)** attributable to the 15 dietary risks under **2017 exposure distributions**.

### Analytical Scenario 2: 

Assesses the change in burden when shifting the mean intake of one or more dietary risks by a fixed amount (the h‑shift; positive = increase, negative = decrease). Shifts can be set per risk, country, age, sex, time. Outputs include:
- Joint dietary risks (J) (disease burden from all dietary exposures - mediation-adjusted across correlated risks)
- Non‑joint disease burden from dietary risks (NJ) (disease burden from single dietary exposures - no mediation)
- Proportional disease burden from dietary risks (PJ) DALYs (disease burden from single dietary exposures – proportional to joint exposure)

The sum of the proportional disease burden from single dietary exposures (PJ) add to the joint disease burden from all dietary exposure (J). The non‑joint disease burden from dietary risks (NJ) are super-additive, they add to more than the joint disease burden from dietary exposure (J).

### Analytical Scenario 3:

Computes **marginal PAFs** and **marginal DALYs** of the joint dietary disease burden around small changes in an intake baseline. Optionally forward-looking using **Shared Socioeconomic Pathways (SSPs)** for exposure means and exogenous YLL/YLD projections.

### Forward -looking analyses (Extrapolations) 

Iterate over **years and scenarios** by providing time varying exposure means and disease burden projections. Indexing is handled centrally to ensure alignment across inputs and outputs.

Schematic diagrams illustrating the workflows of the different analytical scenarios are provided in the 'Schematic Diagrams' subfolder within the 'Additional Information' directory (can also be found in Appendix A of ‘README_extended.pdf’). Further detailed documentation is available in ‘README_extended.pdf’ in the same folder. Additional information on the calculation of dietary mean exposure and disease rate projections is provided in Appendix C of the same .pdf file.

## License & Data Use 

- Code license: MIT (this repository)
- GBD-derived data inputs: Subject to IHME’s Free of Charge Non-Commercial User Agreement. These data are not distributed in the public repo. See the ‘README_extended.pdf’ in the ‘Additional Information’ folder of the repository.

### How to access input data: 

1. Register and accept terms on the IHME website (retain proof/screenshot).
2. Email the authors (see Contact) with proof. 
3. You will receive access to a restricted Zenodo repository containing the input datasets prepared for this emulator. 

Availability of the emulator code does not grant rights to use IHME data for commercial purposes. Commercial users must obtain a separate license from IHME. 

## Requirements

- Python: 3.11.7 (Anaconda recommended)
- Core libraries: numpy, scipy, pandas, sympy, matplotlib

## Repository Structure (high level)

- **Original GBD Emulator/** – Analytical Scenario 1 scripts 
    - Original_GBD.py
- **Unilateral Shift Intake/** – Analytical Scenario 2 scripts 
    - Unilateral_Shift.py (joint/nonjoint via flag)
    - Unilateral_Shift_PJ.py (proportional joint decomposition)
- **Marginals Calculation/** – Analytical Scenario 3 scripts 
    - Partial_Derivative_Calculation.py
- **Data/** – placeholder only (replace with authorized dataset) 
    - Expected subfolders include Shift/, Projections/, SSP Means/, Predictions/ (created on run)
- **Additional Information/**
    - Contains documents that provide detailed explanations of the emulator’s logic, workflows, and implementation. 

Supporting modules (per scenario folders): Setup_file.py, helpers.py, helpers_data_and_setup.py, helpers_variables_calculation.py, Variable_creater_class.py, Distribution_creater_class.py, helpers_PAF_calculation.py.

For more informational on data use license and overview of the modules refer to ‘README_extended.pdf’ in the ‘Additional Information folder’. For a more detailed description of the main scripts and the supporting modules, refer to Appendix B in the same extended .pdf file. 

## Outputs 

Results are written as CSV to Data/Predictions/ subfolders. Typical outputs include:
- Scenario 1: Country-level totals (DALYs) aggregated over risks, outcomes, ages, and sex.
- Scenario 2: DALY changes per risk (joint or non‑joint), with optional proportional joint decomposition to attribute the joint total to individual risks.
- Scenario 3: Marginal DALY changes per risk (by age grouping), for each year and country under the selected SSP.

## Note 

This README provides a high-level overview. For detailed explanations of the emulator’s logic, workflows, and advanced configuration, refer to **README_extended.pdf**. Additionally, refer to the appendices for a comprehensive understanding of the model and its implementation.

## Contact 

For questions regarding the emulator, or to request access to the private code repository and associated data, please contact:

**Dr Steven Lord**  
Senior Researcher Food System Economics 
Food Systems Transformation Group
Environmental Change Institute, University of Oxford  
Email: steven.lord@ouce.ox.ac.uk

**Shaun Solomon**  
Research Programmer and Data Analyst for Food System Economic Cost Modelling 
Food Systems Transformation Group
Environmental Change Institute, University of Oxford  
Email: shaun.solomon@ouce.ox.ac.uk

## Citation 

Please cite the emulator and associated work when using this repository or derivative outputs. A suggested format and BibTeX will be provided in the repository’s CITATION.cff. Also cite the GBD methodology and any specific IHME data products used.

## References 

1.	Global Burden of Disease (GBD). https://www.healthdata.org/research-analysis/gbd
2.	Afshin A. et al. The Lancet (2019). Health effects of dietary risks in 195 countries, 1990–2017.




