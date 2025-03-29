# Scaler Learner Clustering Project

## Overview

This project performs an analysis and clustering of Scaler learners based on their employment data, including company, job position, compensation (CTC), and experience. The analysis includes:

*   Exploratory Data Analysis (EDA)
*   Data Cleaning and Preprocessing
*   Feature Engineering (Years of Experience)
*   Manual Clustering based on relative CTC within peer groups (Company, Job, Experience)
*   Unsupervised Clustering using K-Means (attempted Hierarchical clustering)

The goal is to identify learner segments and derive insights for Scaler's analytics and career services teams.

## Data

The analysis uses the `scaler_clustering.csv` dataset, which contains anonymized learner data.

## Requirements

The analysis is performed using Python and requires the following libraries:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn (`sklearn`)
*   scipy

You can typically install these using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to Run

1.  Ensure you have Python and the required libraries installed.
2.  Place the `scaler_clustering.csv` file in the same directory as the script.
3.  Run the script from your terminal:
    ```bash
    python scaler_analysis.py
    ```

The script will print analysis steps, summaries, and findings to the console.

## Output Files

The script generates the following output files in the project directory:

*   `scaler_analysis.py`: The Python script itself (updated with improvements).
*   `scaler_clustered_data.csv`: The final processed dataset containing cleaned data, engineered features, manual clustering flags, and unsupervised cluster labels.
*   `documentation.md`: Detailed explanation of the project steps, findings, and observations.
*   `readme.md`: This file.
*   **Plots (PNG format):**
    *   `dist_*.png`: Distribution plots for numerical features (`ctc`, `Years_of_Experience`, `ctc_updated_year`).
    *   `count_*.png`: Count plots for top categorical features (`Company_hash_Cleaned`, `Job_position_Cleaned`).
    *   `bivariate_exp_ctc.png`: Scatter plot showing the relationship between Years of Experience and CTC.
    *   `kmeans_elbow_plot.png`: Elbow method plot to help determine the optimal K for K-Means (generated from a sample).
    *   `hierarchical_dendrogram.png`: Attempted dendrogram plot for hierarchical clustering (may be incomplete or missing due to data size).

## Notes

*   The unsupervised K-Means clustering performed on the high-dimensional one-hot encoded data resulted in highly skewed clusters. Further refinement might be needed for more balanced unsupervised segmentation.
*   Hierarchical clustering was limited by computational/memory constraints.
*   The manual clustering flags (`Designation`, `Class`, `Tier`) provide valuable insights into relative compensation.
*   The script includes improvements like robust handling of invalid `orgyear`, capping `Years_of_Experience`, filtering Top N rankings by group size, and using log scales for skewed visualizations.
