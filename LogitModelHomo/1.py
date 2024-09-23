
import numpy as np 
import pandas as pd
import os
import statsmodels.api as sm
from function import cal1,k
from function2 import cal2
df = pd.read_csv(os.path.join(os.getcwd(),'LogitModelHomo\\product_data.csv'))

df['total_market_share'] = df.groupby('market_ids')['shares'].transform('sum')

# Step 2: Calculate "no market share"
df['no_market_share'] = 1 - df['total_market_share']

# Step 3: Calculate the log of product share and no market share
df['log_product_share'] = np.log(df['shares'])
df['log_no_market_share'] = np.log(df['no_market_share'])

# Step 4: Create the new column for log difference
df['log_difference'] = df['log_product_share'] - df['log_no_market_share']


independent_vars = df[['prices','sugar']]

X = sm.add_constant(independent_vars)

# Step 3: Dependent variable (log_difference)
y = df['log_difference']

# Step 4: Run OLS regression
model = sm.OLS(y, X).fit()

# Step 5: Print the summary of the OLS regression results
print(model.summary())


from linearmodels.iv import IV2SLS
endog_var = df['prices']
exog_vars = df[['sugar']]
# iv_vars = df[['distance_to_market', 'supply_chain_quality']]
print(df.head())