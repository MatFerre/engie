
 
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -*- coding: utf-8 -*-

import dataiku

import pandas as pd, numpy as np

from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

import statsmodels

import statsmodels.api as sm

import statsmodels.formula.api as smf

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

import os

import sys

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

# %matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression  # Add this import

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

from datetime import date

import holidays

from workalendar.europe import France

import datetime

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# see what was set in "variables"

dataiku.get_custom_variables()

execution_date = dataiku.get_custom_variables()['execution_date']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# %%time

# Read recipe inputs

LMM = dataiku.Dataset("LMM_bis")

data = LMM.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# ## Adjust pce to 13 numbers

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data['pce']=data['pce'].astype(str)

# data['pce'].apply(len).unique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Define a function to add leading zeros if length is less than or equal to 13

def add_leading_zeros(pce_value):

    if len(pce_value) <= 13:

        # Pad the string with leading zeros to make it 14 characters long

        return pce_value.zfill(14)

    else:

        return pce_value

# Apply the function to the 'pce' column

data['pce'] = data['pce'].apply(add_leading_zeros)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# ### !!! DROP¨!!!###

# unique_values = data['pce'].unique()

# # Ensure there are at least 100 unique values to sample

# if len(unique_values) < 100:

#     raise ValueError("The column does not contain at least 100 unique values.")

# # Randomly select 100 unique values from the column

# selected_values = np.random.choice(unique_values, size=100, replace=False)

# # Subset the DataFrame to include only rows with the selected values

# subset_df = data[data['pce'].isin(selected_values)]

# data = subset_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# ## Trend per pce

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data = data.sort_values(by=['pce', 'gasday'])

data['trend'] = data.groupby('pce').cumcount() + 1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Add workday & week-end

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data['gasday'] = pd.to_datetime(data['gasday']).dt.date

# Function to determine if a day is a workday (Mon-Fri) or weekend (Sat-Sun)

def is_workday(date):

   return 1 if date.weekday() < 5 else 0

# Apply the function to create the Workday column

data['workday'] = data['gasday'].apply(is_workday)

data['weekend'] = 1-data['workday']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# ## Add Holidays

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Assuming col 'year'

unique_years = data['year'].unique()

# Convert unique_years to integers and create a list

unique_years_list = unique_years.astype(int).tolist()

# Create an empty dictionary to store the holidays for each year

fr_holidays = {}

# Iterate through unique years and get the public holidays for each year

for year in unique_years_list:

    fr_holidays[year] = holidays.France(years=[year])

# Flatten the holiday dictionary for easier lookup

all_holidays = {date: name for year in fr_holidays for date, name in fr_holidays[year].items()}

all_holidays


# Create a new column indicating whether each date is a public holiday (1 or 0)

data['Is_Public_Holiday'] = data['gasday'].apply(lambda x: 1 if x in all_holidays else 0)

# Create a column with the holiday name or an empty string if it is not a holiday

data['Holiday_Name'] = data['gasday'].apply(lambda x: all_holidays.get(x, ''))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Determine if it's a non-working day

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data['non_working_day'] = data['weekend'] | data['Is_Public_Holiday']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Count observations per category

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data['count'] = data.groupby('pce')['pce'].transform('count')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Check for missing values and drop rows with missing values

data_nona = data.dropna(subset=['valeur_energie_conso', 'hdd', 'trend', 'pce'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Subset the dataset where column 'is_outlier' is not equal to -1

# data_nona = data_nona[data_nona['is_outlier'] != -1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data_nona.shape

data_nona = data_nona[(data_nona['hdd'] != 0) & (data['valeur_energie_conso'] != 0)]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# FE weights are related to overall residual variance (Captures "leftover" variability within groups after accounting for both fixed and random effects), RE varince (Captures how much group-specific deviations (random effects) vary around the population average (fixed effects).the RE variance-covariance matrix is used to weight the FE estimates and their uncertainty. The FE estimates are derived by "pooling" group-specific effects, with pooling weights determined by:

# The relative magnitude of within-group vs. between-group variance,

# Correlations between random effects (e.g., intercept-slope covariance).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# ## Models

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# #%%time

# lmm = smf.mixedlm(

#     formula='valeur_energie_conso ~ hdd * workday + hdd*non_working_day - workday - non_working_day - hdd + trend + 1',

#     data=data_nona,

#     groups="pce",

#     re_formula="~hdd*workday+hdd*non_working_day - workday - non_working_day - hdd + 1"

# ).fit()

# lmm.summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

#%%time

lmm_default = smf.mixedlm(

    formula='valeur_energie_conso ~ hdd + trend + Is_Public_Holiday',

    data=data_nona,

    groups="pce",

    re_formula="~hdd + 1"

).fit()

lmm_default.summary()

# print(lmm_default.cov_re)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

lmm_default.summary()

# print(lmm_default.cov_re)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Because residuals have heavy tail - try log(conso)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data_nona['log_valeur'] = np.log(data_nona['valeur_energie_conso'])

# lmm_default = smf.mixedlm(

#     formula='log_valeur ~ hdd + trend',

#     data=data_nona,

#     groups="pce",

#     re_formula="~hdd + 1"

# ).fit()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# df_random_effects_default = pd.DataFrame.from_dict(lmm_default.random_effects, orient='index')

# # Add fixed effects to the DataFrame

# df_random_effects_default['Fixed intercept'] = lmm_default.fe_params.loc['Intercept']

# df_random_effects_default['Fixed slope trend'] = lmm_default.fe_params.loc['trend']

# df_random_effects_default['Fixed slope Is_Public_Holiday'] = lmm_default.fe_params.loc['Is_Public_Holiday']

# df_random_effects_default['Fixed slope hdd'] = lmm_default.fe_params.loc['hdd']

# # Rename columns for clarity

# df_random_effects_default = df_random_effects_default.rename({

#     'pce': 'Random intercept',

#     'hdd': 'Random slope hdd'

# }, axis=1)

# df_random_effects_default.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# ###  When to Examine Individual Contributions:

# Understanding Variability:

# 

# If you're interested in understanding the sources of variation in your data, breaking down the residuals into fixed and random components can provide insights into how much variation is explained by the fixed effects versus the random effects.

# Model Diagnostics:

# 

# Examining separate residuals can help diagnose potential issues with model fit. For instance, large random effects residuals might suggest that the random structure is not adequately capturing the variability in the data.

# Random Effects Interpretation:

# 

# In hierarchical data, understanding the impact of random effects (e.g., variability between groups) can be crucial. Individual residuals can highlight how each group or subject deviates from the overall model prediction.

# Improving Model Specification:

# 

# If fixed effects residuals are large or patterned, it might indicate that additional fixed effects should be considered. Similarly, unexpected patterns in random effects residuals could suggest a need to reconsider the random structure.

# 

# ### When Common Residuals Might Be Sufficient:

# Simpler Analysis:

# 

# If your primary goal is straightforward prediction or if the model fits well, focusing on total residuals might suffice, as they provide a general measure of model error.

# Overall Model Fit:

# 

# When the primary concern is the overall predictive accuracy of the model, common residuals (total residuals) may provide enough insight to assess model performance.

# Limited Complexity:

# 

# In cases where the random effects structure is simple or the variability between groups is not a primary interest, detailed residual analysis might not be necessary.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# import pandas as pd

# import statsmodels.api as sm

# # Assuming df_random_effects_default and lmm_default are already defined

# # Extract the relevant columns from the original dataset

# predict_data = data_nona[['pce', 'hdd', 'trend', 'Is_Public_Holiday']]

# predict_data.set_index('pce', inplace=True)

# # Total residuals from the model

# total_residuals = lmm_default.resid

# total_residuals.index = predict_data.index


# # Add a constant for the intercept

# predict_data = sm.add_constant(predict_data)

# predict_data.rename(columns={'const': 'Intercept'}, inplace=True)

# # Ensure predict_data columns match the fixed effects' index

# predict_data = predict_data[lmm_default.fe_params.index]

# # Calculate predictions using only the fixed effects

# fixed_effects_predictions = predict_data.dot(lmm_default.fe_params)

# # Align the actual values by setting 'pce' as the index

# actual_values = data_nona.set_index('pce')['valeur_energie_conso']

# # Calculate residuals: actual - fixed effects predictions

# residuals_fixed_effects = actual_values - fixed_effects_predictions

# # Calculate random effects residuals: total residuals - fixed effects residuals

# residuals_random_effects = total_residuals - residuals_fixed_effects

# # Create a DataFrame with the residuals

# residuals_df = pd.DataFrame({

#     'Total Residuals': total_residuals,

#     'Fixed Effects Residuals': residuals_fixed_effects,

#     'Random Effects Residuals': residuals_random_effects

# })

# # Set 'pce' as the index

# residuals_df.index = actual_values.index

# # Display the DataFrame

# print(residuals_df.head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# dates=data_nona[['gasday']]

# dates.index = predict_data.index

# residuals_df[['gasday']]=dates

# residuals_df = residuals_df.reset_index()

# residuals_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # save residuals to a table :

# # ntile100 = dataiku.get_custom_variables()['ntile100']

# # ntile100

# LMM_residuals = dataiku.Dataset("LMM_residuals")

# # LMM_residuals.set_write_partition(ntile100)

# LMM_residuals.write_with_schema(residuals_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Analysis of the residuals for lateron

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# fitted_values_summary = lmm_default.fittedvalues.describe()

# print(fitted_values_summary)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# import matplotlib.pyplot as plt

# import statsmodels.api as sm

# import seaborn as sns

# import scipy.stats as stats

# # Assuming lmm_default is your fitted model object

# # Adjust the figure size for the Q-Q plot

# plt.figure(figsize=(12, 6))

# # Use scipy's probplot for more control over the plot

# residuals = lmm_default.resid  # Ensure this is correctly defined

# stats.probplot(residuals, dist="norm", plot=plt)

# plt.title('Q-Q Plot of Residuals')

# plt.grid(True)

# plt.show()

# # Adjust the figure size for the histogram/KDE plot

# plt.figure(figsize=(12, 6))

# sns.histplot(residuals, kde=True)

# plt.title('Residual Distribution')

# plt.grid(True)

# plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # Assuming residuals_df is already defined and is a DataFrame

# df = pd.DataFrame(residuals_df)

# # Randomly select 5 unique 'pce' values from the index

# selected_pce = np.random.choice(df.index.unique(), 1, replace=False)

# # selected_pce = np.random.choice(df.index.unique(), residuals_df.index.nunique(), replace=False)

# # Plotting residuals for each selected 'pce'

# plt.figure(figsize=(12, 8))

# for pce in selected_pce:

#     subset = df.loc[pce]  # Select the subset of data for the current 'pce'

# #     plt.scatter(subset['gasday'], subset['Total Residuals'], label=f'Total Residuals - {pce}', marker='o')

# #     plt.scatter(subset['gasday'], subset['Fixed Effects Residuals'], label=f'Fixed Effects Residuals - {pce}', marker='o')

# #     plt.scatter(subset['gasday'], subset['Random Effects Residuals'], label=f'Random Effects Residuals - {pce}', marker='o')

# # Plotting residuals for the current 'pce' with smaller asterisk markers

#     plt.scatter(subset['gasday'], subset['Total Residuals'], label='Total Residuals', marker='o')  # Smaller size for total residuals

#     plt.scatter(subset['gasday'], subset['Fixed Effects Residuals'], label='Fixed Effects Residuals', edgecolors='g',facecolors='none', s=25)  # Asterisk for FE

#     plt.scatter(subset['gasday'], subset['Random Effects Residuals'], label='Random Effects Residuals', edgecolors='c',facecolors='none', s=25)  # Asterisk for RE

# # Adding labels and title

# plt.xlabel('Date')

# plt.ylabel('Residuals')

# plt.title('Residuals for Random PCEs')

# plt.legend(loc='best', fontsize='small', ncol=2)

# plt.grid(True)

# # Display the plot

# plt.xticks(rotation=45)  # Rotate date labels for better readability

# plt.tight_layout()  # Adjust layout to fit everything

# plt.show()

# #01123009969958

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# import pandas as pd

# import numpy as np

# import matplotlib.pyplot as plt

# import os

# from matplotlib.backends.backend_pdf import PdfPages

# import dataiku

# # Assuming residuals_df is already defined and is a DataFrame

# df = pd.DataFrame(residuals_df)

# # Randomly select unique 'pce' values from the index

# selected_pce = np.random.choice(df.index.unique(), residuals_df.index.nunique(), replace=False)

# # print(selected_pce)

# title_str = "resid_log"

# # Path setup

# main_folder_path = dataiku.Folder("UQxMDX2I").get_path()

# subfolder_name = "residuals/"

# file = f'{title_str}.pdf'  # Output PDF file

# output_path = os.path.join(main_folder_path, subfolder_name, file)

# # Make sure the directory exists

# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# # Create a PDF file to save multiple plots

# with PdfPages(output_path) as pdf:

#     for pce in selected_pce:

#         plt.figure(figsize=(12, 8))

#         # Select the subset of data for the current 'pce'

#         subset = df.loc[pce]

#         # Plotting residuals for the current 'pce'

# #         plt.scatter(subset['gasday'], subset['Total Residuals'], label='Total Residuals', marker='o')

# #         plt.scatter(subset['gasday'], subset['Fixed Effects Residuals'], label='Fixed Effects Residuals', marker='o')

# #         plt.scatter(subset['gasday'], subset['Random Effects Residuals'], label='Random Effects Residuals', marker='o')


#         # Plotting residuals for the current 'pce' with smaller asterisk markers

#         plt.scatter(subset['gasday'], subset['Total Residuals'], label='Total Residuals', marker='o')  # Smaller size for total residuals

#         plt.scatter(subset['gasday'], subset['Fixed Effects Residuals'], label='Fixed Effects Residuals', edgecolors='g',facecolors='none', s=25)  # Asterisk for FE

#         plt.scatter(subset['gasday'], subset['Random Effects Residuals'], label='Random Effects Residuals', edgecolors='c',facecolors='none', s=25)  # Asterisk for RE

#         # Adding labels and title

#         plt.xlabel('Date')

#         plt.ylabel('Residuals')

#         plt.title(f'Residuals for PCE: {pce}')

#         plt.legend(loc='best', fontsize='small', ncol=2)

#         plt.grid(True)

#         # Adjust layout to fit everything

#         plt.xticks(rotation=45)

#         plt.tight_layout()

#         # Save the current plot to the PDF

#         pdf.savefig()

#         plt.close()

# # Display confirmation

# print(f"All plots saved to {output_path}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# limited_results = {}

# count = 0

# for pce, val in lmm.random_effects.items():

#     if pce in data_nona['pce'].unique():

#         limited_results[pce] = val

#         count += 1

#         if count == 5:

#             break

# # print(limited_results)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# df_random_effects = pd.DataFrame.from_dict(lmm.random_effects, orient='index')

# df_random_effects = df_random_effects.rename({'pce':'Fixed intercept','hdd':'Fixed slope','trend':'Fixed trend', 'pce': 'Random intercept', 'hdd': 'Random slope'}, axis=1)

# df_random_effects

# df_random_effects['Fixed intercept'] = lmm.fe_params.loc['Intercept']

# df_random_effects['Fixed slope'] = lmm.fe_params.loc['hdd']

# df_random_effects['Fixed trend'] = lmm.fe_params.loc['trend']

# df_random_effects['INTERCEPT LMM'] = df_random_effects['Random intercept'] + lmm.fe_params.loc['Intercept']

# df_random_effects['SLOPE LMM'] = df_random_effects['Random slope'] + lmm.fe_params.loc['hdd']

# df_random_effects.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# # Get RE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# df_random_effects = pd.DataFrame.from_dict(lmm.random_effects, orient='index')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

df_random_effects_default = pd.DataFrame.from_dict(lmm_default.random_effects, orient='index')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# # Add params of FE to RE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # Add fixed effects to the DataFrame

# df_random_effects['Fixed intercept'] = lmm.fe_params.loc['Intercept']

# df_random_effects['Fixed slope trend'] = lmm.fe_params.loc['trend']

# df_random_effects['Fixed slope hdd * workday'] = lmm.fe_params.loc['hdd:workday']

# df_random_effects['Fixed slope hdd * non_working_day'] = lmm.fe_params.loc['hdd:non_working_day']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # Rename columns for clarity

# df_random_effects = df_random_effects.rename({

#     'pce': 'Random intercept',

#     'hdd:workday': 'Random slope hdd * workday',

#     'hdd:non_working_day': 'Random slope hdd * non_working_day',

# }, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Add fixed effects to the DataFrame

df_random_effects_default['Fixed intercept'] = lmm_default.fe_params.loc['Intercept']

df_random_effects_default['Fixed slope trend'] = lmm_default.fe_params.loc['trend']

df_random_effects_default['Fixed slope Is_Public_Holiday'] = lmm_default.fe_params.loc['Is_Public_Holiday']

df_random_effects_default['Fixed slope hdd'] = lmm_default.fe_params.loc['hdd']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Rename columns for clarity

df_random_effects_default = df_random_effects_default.rename({

    'pce': 'Random intercept',

    'hdd': 'Random slope hdd'

}, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

# # Recalculate individual slopes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # Calculate the combined (fixed + random) effects

# df_random_effects['INTERCEPT LMM'] = df_random_effects['Random intercept'] + df_random_effects['Fixed intercept']

# df_random_effects['SLOPE LMM hdd * workday'] = df_random_effects['Random slope hdd * workday'] + df_random_effects['Fixed slope hdd * workday']

# df_random_effects['SLOPE LMM hdd * non_working_day'] = df_random_effects['Random slope hdd * non_working_day'] + df_random_effects['Fixed slope hdd * non_working_day']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Calculate the combined (fixed + random) effects

df_random_effects_default['INTERCEPT LMM'] = df_random_effects_default['Random intercept'] + df_random_effects_default['Fixed intercept']

df_random_effects_default['SLOPE LMM hdd'] = df_random_effects_default['Random slope hdd'] + df_random_effects_default['Fixed slope hdd']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# delete id needed

df_random_effects = df_random_effects_default

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# ADD OLS

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# # Dictionary to store OLS results

# ols_results = {'INTERCEPT OLS': {}, 'SLOPE OLS': {}}

# # Perform OLS for each group

# for group in data['pce'].unique():

#     group_data = data[data['pce'] == group]

#     X = sm.add_constant(group_data['hdd'])

#     y = group_data['valeur_energie_conso']

#     ols_model = sm.OLS(y, X).fit()

#     # Print the parameters to verify names

# #     print(f"Parameters for group {group}:", ols_model.params)

#     ols_results['INTERCEPT OLS'][group] = ols_model.params[0]  # Assuming the first parameter is the intercept

#     ols_results['SLOPE OLS'][group] = ols_model.params['hdd']

# # Convert OLS results to DataFrame

# df_ols_results = pd.DataFrame.from_dict(ols_results)

# # Merge OLS results with the random effects DataFrame

# df_random_effects = df_random_effects.join(df_ols_results)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# test = df_random_effects.head(30)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

df_random_effects = df_random_effects.reset_index().rename(columns={'index': 'pce'})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# df_random_effects_default = df_random_effects_default.reset_index().rename(columns={'index': 'pce'})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

df_random_effects

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Rename columns in df2 if they are the same as in df1

df_random_effects_default.columns = [col + '_default' if col in df_random_effects.columns else col for col in df_random_effects_default.columns]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

df_random_effects

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Merge the dataframes on 'pce'

# merged_df = pd.merge(df_random_effects, df_random_effects_default, left_on='pce', right_on='pce_default', how='outer')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# merged_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Drop the duplicate 'pce_default' column, if necessary

# merged_df = merged_df.drop(columns=['pce_default'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

merged_df = df_random_effects

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data_nona.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

data_nona_unique = data_nona[['pce', 'count']].drop_duplicates(subset='pce')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data_nona_unique

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Convert 'pce' column in both DataFrames to string

merged_df['pce'] = merged_df['pce'].astype(str)

data_nona_unique['pce'] = data_nona_unique['pce'].astype(str)

merged_df=pd.merge(merged_df, data_nona_unique[['pce', 'count']], on='pce', how='left')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# merged_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# merged_df.sample(15)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# test

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data[['pce']].count()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# data_nona[['pce']].count()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# random_row_indices = df_random_effects.index.to_series().sample(n=3, random_state=8).tolist()

# random_row_indices

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# df_random_effects.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# import pandas as pd

# import seaborn as sns

# import matplotlib.pyplot as plt

# # List of 'pce' values to iterate over

# # pce_ids = [14450217022650, 6463531105264, 23248625112754]  # Example list, add more pce_ids as needed

# pce_ids = random_row_indices

# # Set plot style

# sns.set_style('whitegrid')

# # Create the plot

# plt.figure(figsize=(14, 6))

# # Loop over each pce_id

# for pce_id in pce_ids:

#     # Extract fixed effects (alpha and beta) from the model

#     alpha = df_random_effects.loc[pce_id, 'INTERCEPT LMM']

#     beta = df_random_effects.loc[pce_id, 'SLOPE LMM']

#     alpha_ols = df_random_effects.loc[pce_id, 'INTERCEPT OLS']

#     beta_ols = df_random_effects.loc[pce_id, 'SLOPE OLS']

#     # Filter the data for the specific 'pce' value

#     pce_data = data[data['pce'] == pce_id]

#     # Scatter plot of the data points

#     sns.scatterplot(data=pce_data, x='hdd', y='valeur_energie_conso', alpha=.5, s=60, label=f'PCE {pce_id}')

#     # Plot the regression line

#     hdd_range = pd.Series([pce_data['hdd'].min(), pce_data['hdd'].max()])

#     plt.plot(hdd_range, alpha + beta * hdd_range, label=f'Regression Line PCE {pce_id}')

#     plt.plot(hdd_range, alpha_ols + beta_ols * hdd_range, label=f'Regression Line OLS PCE {pce_id}')

# # Set plot title and labels

# plt.title('Raw Data with Regression Lines for Multiple PCEs')

# plt.xlabel('HDD')

# plt.ylabel('Valeur Energie Conso')

# # Show legend

# plt.legend()

# # Show the plot

# plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

merged_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Write recipe outputs

LLM_Gradient = dataiku.Dataset("LLM_Gradient")

LLM_Gradient.write_with_schema(merged_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Get today's date ==> Set it to execution date;

today_date = datetime.datetime.now().strftime("%Y-%m-%d")

today_date = execution_date

target_partition = dataiku.get_custom_variables()['ntile100']

file_name = f"{target_partition}_coefs_{today_date}.csv"

file_name


# Get the path of the main folder

main_folder_path = dataiku.Folder("UQxMDX2I").get_path()

# Define the subfolder name

subfolder_name = "coefs/"

subfolder_name_bis = f"coefs_{execution_date}"

# Combine the paths using os.path.join

subfolder_path = os.path.join(main_folder_path, subfolder_name, subfolder_name_bis)

# Vérifier si le sous-dossier n'existe pas déjà et le créer si nécessaire

if not os.path.exists(subfolder_path):

    os.makedirs(subfolder_path)

    print(f"Le sous-dossier '{subfolder_name_bis}' a été créé avec succès.")

else:

    print(f"Le sous-dossier '{subfolder_name_bis}' existe déjà.")

output_path = os.path.join(subfolder_path, file_name)

output_path


merged_df.to_csv(output_path, index=False)
 