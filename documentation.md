# Scaler Learner Clustering Analysis - Documentation

## 1. Project Objective

The goal of this project is to analyze a dataset of Scaler learners to identify distinct clusters based on their job profiles, company affiliations, compensation (CTC), and experience. This involves both manual clustering based on predefined rules and unsupervised machine learning techniques (K-Means, Hierarchical). The ultimate aim is to provide actionable insights for Scaler's analytics vertical regarding career paths, company profiles, and potential areas for learner support or curriculum development.

## 2. Data

*   **Source File:** `scaler_clustering.csv`
*   **Data Dictionary:**
    *   `Unnamed: 0`: Index column (dropped during preprocessing).
    *   `company_hash`: Anonymized identifier for the learner's employer.
    *   `email_hash`: Anonymized identifier for the learner.
    *   `orgyear`: Year the learner started employment at the company.
    *   `ctc`: Current Cost To Company (compensation).
    *   `job_position`: Learner's job title/role.
    *   `ctc_updated_year`: Year the CTC was last updated.

## 3. Analysis Steps (`scaler_analysis.py`)

### 3.1. Data Loading and Initial EDA

*   Loaded the dataset using pandas (`low_memory=False` used for robustness).
*   Performed initial exploration:
    *   Checked data shape, column types, and non-null counts (`.info()`).
    *   Examined the first few rows (`.head()`).
    *   Generated statistical summaries for numerical (`.describe()`) and categorical columns (`.describe(include='object')`).
    *   Checked for duplicate `email_hash` entries (indicating multiple records per learner or data issues).
    *   Checked unique counts and top values for `company_hash` and `job_position`.
*   **Observations:** Missing values identified in `company_hash`, `orgyear`, `job_position`. Data quality issues noted in `orgyear` (invalid years like 0, 20165) and `ctc` (potential extreme outliers like 1e9).

### 3.2. Data Preprocessing

*   **Dropped Index:** Removed the `Unnamed: 0` column.
*   **Handled Missing Values:**
    *   Filled missing `company_hash` and `job_position` with placeholders ('Unknown_Company', 'Unknown_Position').
*   **Cleaned Text Columns:**
    *   Created `Company_hash_Cleaned` and `Job_position_Cleaned` by:
        *   Removing non-alphanumeric characters (except whitespace).
        *   Converting to lowercase.
        *   Stripping leading/trailing whitespace.
        *   Consolidating multiple whitespaces.
        *   Replacing any resulting empty strings with 'unknown_company'/'unknown_position'.
*   **Processed Numerical Columns:**
    *   Converted `ctc`, `ctc_updated_year`, `orgyear` to numeric types, coercing errors.
    *   Handled invalid `orgyear` values (outside `MIN_VALID_YEAR` [1960] - `CURRENT_YEAR`) by setting them to NaN before imputation.
    *   Imputed remaining missing `ctc` (if any) using the median.
    *   Imputed remaining missing `orgyear` using `KNNImputer` (k=5) based on `ctc` after scaling. Ensured imputed `orgyear` is integer.
*   **Removed Duplicates:** Dropped duplicate rows based on all columns.
*   **Feature Engineering:**
    *   Created `Years_of_Experience` = Current Year - `orgyear`.
    *   Capped `Years_of_Experience` between 0 and `MAX_YEARS_EXPERIENCE` (60 years) to handle outliers from `orgyear`.

### 3.3. Further EDA (Post-Preprocessing)

*   Generated and saved plots:
    *   Histograms for numerical features (`ctc` [log-scaled], `Years_of_Experience`, `ctc_updated_year`) - saved as `dist_*.png`.
    *   Bar plots for the top 20 most frequent cleaned companies and job positions - saved as `count_*.png`.
    *   Scatter plot of `Years_of_Experience` vs `ctc` (log-scaled y-axis, sampled) - saved as `bivariate_exp_ctc.png`.

### 3.4. Manual Clustering

*   Calculated average (mean), median, min, max CTC, and count grouped by:
    1.  `Company_hash_Cleaned`, `Job_position_Cleaned`, `Years_of_Experience`
    2.  `Company_hash_Cleaned`, `Job_position_Cleaned`
    3.  `Company_hash_Cleaned`
*   Merged these summary statistics back into the main DataFrame.
*   Created three flags based on comparing an individual's CTC to the relevant group average:
    *   **`Designation` Flag:** Compares CTC to the average for the *same Company, Job Position, and Years of Experience*. (1: Above Avg, 2: Equal/No Avg/NaN, 3: Below Avg)
    *   **`Class` Flag:** Compares CTC to the average for the *same Company and Job Position*. (1: Above Avg, 2: Equal/No Avg/NaN, 3: Below Avg)
    *   **`Tier` Flag:** Compares CTC to the average for the *same Company*. (1: Above Avg, 2: Equal/No Avg/NaN, 3: Below Avg)

### 3.5. Answering Questions (Manual Clustering)

*   Identified and printed:
    *   Top 10 highest earners relative to their company average (Tier 1).
    *   Bottom 10 lowest earners relative to their company average (Tier 3).
    *   Top/Bottom 10 Data Science role earners relative to their company/job average (Class 1 / Class 3).
    *   Top 10 earners for a specific role ('software engineer') and experience (5 years) relative to their company/job/experience average (Designation 1).
    *   Top 10 companies by average CTC (considering only companies with >= `MIN_GROUP_SIZE_FOR_RANKING` [10] employees).
    *   Top 2 positions per company by average CTC (considering only company/position groups with >= `MIN_GROUP_SIZE_FOR_RANKING` [10] employees).

### 3.6. Data Processing for Unsupervised Clustering

*   Selected features: `ctc`, `Years_of_Experience`, `Company_hash_Cleaned`, `Job_position_Cleaned`.
*   Dropped rows with any remaining NaNs in these selected features.
*   Separated numerical and categorical features.
*   Applied `OneHotEncoder` to categorical features, generating a **sparse matrix** to handle high dimensionality efficiently.
*   Combined the numerical features (as a dense array) and the sparse encoded categorical features using `scipy.sparse.hstack`.
*   Standardized the combined sparse matrix using `StandardScaler(with_mean=False)`, necessary for sparse data.

### 3.7. Unsupervised Clustering

*   **K-Means:**
    *   Performed the Elbow Method on a sample (`SAMPLE_SIZE`) of the scaled sparse data to help determine an optimal K (plot saved as `kmeans_elbow_plot.png`). Recommended manual inspection of the plot to potentially adjust `OPTIMAL_K`.
    *   Applied `KMeans` (using `OPTIMAL_K`, default 5) to the **full** scaled sparse dataset.
    *   Added cluster labels (`KMeans_Cluster`) back to the original DataFrame.
    *   **Observation:** The resulting clusters were highly skewed, with one cluster containing the vast majority of data points.
*   **Hierarchical Clustering:**
    *   Attempted linkage calculation and dendrogram plotting on the sample (truncated for readability). This often failed or produced limited results due to memory constraints or limitations in handling high-dimensional sparse data (plot saved as `hierarchical_dendrogram.png` if successful).
    *   Attempted `AgglomerativeClustering` on the sample, converting to dense only if memory estimate was below a threshold. Often skipped due to size.
    *   Added cluster labels (`Hierarchical_Cluster`) back to the original DataFrame (value is -1 if clustering failed or row wasn't sampled).
    *   **Observation:** Hierarchical clustering is computationally intensive and memory-limited for this dataset size and dimensionality.

### 3.8. Insights and Output

*   Analyzed K-Means cluster characteristics (mean numerical values, top categorical values per cluster). Noted the skewness.
*   Saved the final processed DataFrame (including cleaned data, features, manual flags, and cluster labels, with floats rounded) to `scaler_clustered_data.csv`.
*   Printed actionable insights and recommendations based on both manual flag analysis and the (limited) unsupervised clustering results.

## 4. Key Findings & Observations

*   The dataset contains significant numbers of learners, companies, and job positions.
*   Data cleaning was necessary for company/job names and handling invalid/missing `orgyear` values.
*   Extreme outliers exist in the `ctc` column, affecting averages and potentially distance-based clustering. Log-scaling helped visualization but wasn't applied prior to clustering in this iteration.
*   Manual clustering flags (Designation, Class, Tier) provide useful context about relative compensation within specific peer groups.
*   Unsupervised clustering (K-Means) on the one-hot encoded sparse data resulted in highly unbalanced clusters, limiting their practical use for segmentation in this run. This is common with high-dimensional sparse data and K-Means.
*   Hierarchical clustering was largely infeasible due to computational and memory constraints.
*   Top company/position rankings are more reliable when filtered by a minimum group size (`MIN_GROUP_SIZE_FOR_RANKING`).

## 5. Output Files

*   `scaler_analysis.py`: The Python script performing the analysis.
*   `scaler_clustered_data.csv`: The final dataset with all preprocessing, features, flags, and cluster labels.
*   `documentation.md`: This file.
*   `readme.md`: Project overview and setup instructions.
*   **Plots (PNG format):**
    *   `dist_*.png`: Histograms of numerical features.
    *   `count_*.png`: Bar plots of top categorical features.
    *   `bivariate_exp_ctc.png`: Scatter plot of Experience vs CTC.
    *   `kmeans_elbow_plot.png`: Elbow method plot for K-Means (from sample).
    *   `hierarchical_dendrogram.png`: Dendrogram plot (attempted, may be incomplete/missing).

## 6. Future Work Suggestions

*   Implement robust outlier handling for `ctc` (e.g., capping, removal, or log transformation *before* clustering).
*   Explore dimensionality reduction techniques (PCA, TruncatedSVD) on the sparse encoded features before clustering.
*   Experiment with different clustering algorithms more suited for high-dimensional sparse data (e.g., DBSCAN, Birch).
*   Refine feature selection/engineering for unsupervised clustering (e.g., using TF-IDF for text features instead of pure OneHotEncoding if appropriate, feature hashing).
*   Perform deeper analysis of the characteristics of learners within each manual flag category (Tier 1/2/3, Class 1/2/3).
*   Manually inspect the elbow plot (`kmeans_elbow_plot.png`) and potentially re-run with an updated `OPTIMAL_K`.
