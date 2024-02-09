import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def forward_selection(X, y, criterion='bic', significance_level=0.05):
    initial_variables = X.columns.tolist()
    best_variables = []
    best_model = None
    while len(initial_variables) > 0:
        remaining_variables = list(set(initial_variables) - set(best_variables))
        criterion_values = {}
        for candidate in remaining_variables:
            X_c = sm.add_constant(X[best_variables + [candidate]])
            model = sm.OLS(y, X_c).fit()
            if criterion == 'bic':
                criterion_values[candidate] = model.bic
            elif criterion == 'adjusted_r_squared':
                criterion_values[candidate] = model.rsquared_adj
        if criterion_values:
            best_candidate = min(criterion_values, key=criterion_values.get) if criterion == 'bic' else max(criterion_values, key=criterion_values.get)
            best_variables.append(best_candidate)
            if criterion == 'bic':
                best_model = sm.OLS(y, sm.add_constant(X[best_variables])).fit()
            elif criterion == 'adjusted_r_squared':
                best_model = sm.OLS(y, sm.add_constant(X[best_variables])).fit()
            max_p_value = best_model.pvalues[1:].max()
            if max_p_value > significance_level:
                worst_variable = best_model.pvalues[1:].idxmax()
                best_variables.remove(worst_variable)
            else:
                break
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

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Example usage for BIC
best_variables_bic = forward_selection(X, y, criterion='bic')
# Define X_train before using it
# Fit final models based on the selected variables
X_train_bic = sm.add_constant(X_train[best_variables_bic])  # Add constant for intercept
model_bic = sm.OLS(y_train, X_train_bic).fit()

# Calculate MSE for BIC
mse_bic = np.mean((model_bic.predict(X_train_bic) - y_train) ** 2)

# Define best_variables_aic by calling forward_selection function
best_variables_aic = forward_selection(X, y, criterion='adjusted_r_squared')
X_train_aic = sm.add_constant(X_train[best_variables_aic])  # Add constant for intercept
model_aic = sm.OLS(y_train, X_train_aic).fit()
mse_aic = np.mean((model_aic.predict(X_train_aic) - y_train) ** 2)  # Calculate MSE for AIC

print("MSE for the model selected based on AIC:", mse_aic)

# Identify the best model
if mse_bic < mse_aic:
    print("Model selected based on BIC is the best.")
else:
    print("Model selected based on AIC is the best.")
# Output:
# Best variables based on BIC: ['displacement', 'weight', 'cylinders']
# Best variables based on Adjusted R-squared: ['displacement', 'weight', 'cylinders']
# MSE for the model selected based on BIC: 0.020
# MSE for the model selected based on AIC: 0.021
# Model selected based on BIC is the best.
