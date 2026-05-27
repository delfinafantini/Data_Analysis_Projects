
# Data Curation and Imputation — Melbourne Housing Dataset
 
> Practical work for the **Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones** (2026 edition)
 
## Overview
 
This notebook performs a full data curation and preprocessing pipeline on the Melbourne housing dataset (`melb_data.csv`), which contains records of properties sold in Melbourne, Australia. The goal is to produce a clean, encoded, imputed, and dimensionality-reduced matrix ready for machine learning models.
 
---
## Pipeline

 
### 1. Filtering Data
- Dropped `BuildingArea` and `YearBuilt` initially (high proportion of missing values — handled later via imputation).
- Removed `CouncilArea` (~10% missing, redundant with other location variables).
- Dropped rows where `Car` was null (~0.46% of data — too few to impute meaningfully).
- Removed rows where `Bathroom = 0` or `Bathroom > Rooms` (physically inconsistent).
- Converted `Landsize = 0` to `NaN` for later imputation.
- Dropped `Bedroom2` (external source, potential bias, inconsistencies with `Rooms`).
### 2. Numeric Variables
- Converted `Postcode` to categorical (not meaningful as a number).
- Built a numeric NumPy matrix (`matrix_num`).
### 3. Categorical Variables
- Extracted date features: split `Date` into `Year` and `Month`, then moved them to the numeric matrix.
- Analyzed unique category counts per variable.
- Dropped `Address` (too many unique values for One-Hot Encoding) and `Postcode`.
- Reduced high-cardinality variables (`Suburb`, `SellerG`): categories with less than 1% frequency were grouped into `"Other"`.
- Applied **One-Hot Encoding** to: `Type`, `Method`, `Regionname`, `SellerG`, `Suburb`.
### 4. Imputation
- Re-incorporated `BuildingArea` and `YearBuilt` for imputation.
- Filtered outliers using the 1st and 99th percentiles before imputing.
- Applied **StandardScaler** before imputation and inverted afterward.
- Used **IterativeImputer with KNeighborsRegressor** (k=5, max_iter=20) for `Landsize`, `BuildingArea`, and `YearBuilt`.
- Compared against **SimpleImputer (median strategy)** — KNN better preserves the original distributions.
### 5. Dimensionality Reduction (PCA)
- Applied **PCA** (up to 20 components) on the standardized final matrix.
- Analyzed explained variance with a scree plot.
- Added **PC1** and **PC2** to the final matrix as additional features.
- Visualized PCA projection colored by `Price`.
- Mapped property locations geographically using `contextily` + OpenStreetMap.
### 6. Final Output
- Combined numeric, encoded categorical, imputed, and PCA features into a single DataFrame.
- Exported to `melb_data_processed.csv`.
---
## Technologies
 
- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `missingno` — missing data visualization
- `scikit-learn` — encoding, imputation, scaling, PCA
- `contextily` — geographic basemap
---
