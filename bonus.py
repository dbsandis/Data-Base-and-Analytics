# bonus.py
import statsmodels.api as sm
import numpy as np
import pandas as pd #import package

def forward_selection(X, y, significance_level=0.05):
    initial_variables = X.columns.tolist()
    best_variables = []
    while len(initial_variables) > 0:
        remaining_variables = list(set(initial_variables) - set(best_variables))
        new_pval = pd.Series(index=remaining_variables, dtype=float)
        for candidate in remaining_variables:
            X_c = sm.add_constant(X[best_variables + [candidate]])
            model = sm.OLS(y, X_c).fit()
            new_pval[candidate] = model.pvalues[candidate]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_variables.append(new_pval.idxmin())
        else:
            break
    return best_variables




#load the csv file
df = pd.read_csv('auto_mpg_edit.csv') 

# drop the 'car name' column because it is not useful
df.drop(columns=['car name'], inplace=True) 


# Convert the 'horsepower' column to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')


# Drop the rows with missing values
df.dropna(inplace=True)


#'mpg' has a somewhat longish tail and is not precisely normally distributed
# so we will take a log transformation, ( use df['lmpg'] = df['mpg'].apply(np.log) )
df['lmpg'] = df['mpg'].apply(np.log)

# Assuming df is your DataFrame prepared with 'lmpg'
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df['lmpg']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

best_variables_aic = forward_selection(X, y)
print("Best variables based on AIC:", best_variables_aic)
