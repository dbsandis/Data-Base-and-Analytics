import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DEBUG = False

# Load the CSV file
data = pd.read_csv('auto_mpg_edit.csv') 

# Drop the 'car name' column because it is not useful
data.drop(columns=['car name'], inplace=True) 

# Log-transform 'mpg' and define predictors and the response variable
data['lmpg'] = np.log(data['mpg'])

# Example selected_columns based on AIC criteria, adjust accordingly
selected_columns = ['weight', 'horsepower', 'cylinders']  

# Add constant for intercept in model training and predictions
X = sm.add_constant(data[selected_columns])

X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = data['lmpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if DEBUG:print(X_train, X_test, y_train, y_test)

# Stepwise regression function
def stepwise_selection(X, y, criteria):
    selected_columns = []
    best_model = None
    best_mse = np.inf
    if DEBUG:print(X, y, criteria)
    while True:
        candidate_columns = [col for col in X.columns if col not in selected_columns]
        if DEBUG:print('candidate_columns:',candidate_columns)
        if len(candidate_columns) == 0:
            break
        
        best_candidate = None
        for col in candidate_columns:
            if DEBUG:print('col:',col)
            model_columns = selected_columns + [col]
            X_subset = X[model_columns]
            model = sm.OLS(y, X_subset).fit()
            if DEBUG:print('model:',model)
            if criteria == 'AIC':
                if best_candidate is None or model.aic < best_candidate.aic:
                    best_candidate = model
            elif criteria == 'BIC':
                if best_candidate is None or model.bic < best_candidate.bic:
                    best_candidate = model
            elif criteria == 'Adj_R2':
                if best_candidate is None or model.rsquared_adj > best_candidate.rsquared_adj:
                    best_candidate = model
        
        selected_columns.append(best_candidate.params.index[-1])
        if mean_squared_error(y_test, best_candidate.predict(X_test[selected_columns])) < best_mse:
            best_mse = mean_squared_error(y_test, best_candidate.predict(X_test[selected_columns]))
            best_model = best_candidate
            
    return best_model, selected_columns

# Perform stepwise regression with different criteria
criteria_list = ['AIC', 'BIC', 'Adj_R2']
best_models = {}
for criteria in criteria_list:
    best_model, selected_columns = stepwise_selection(X_train, y_train, criteria)
    best_models[criteria] = (best_model, selected_columns)

# Select the best model based on the lowest MSE on the test set
best_criteria = None
best_mse = np.inf
for criteria, (model, selected_columns) in best_models.items():
    mse = mean_squared_error(y_test, model.predict(X_test[selected_columns]))
    if mse < best_mse:
        best_criteria = criteria
        best_mse = mse

best_model, selected_columns = best_models[best_criteria]

# Print the best model and selected features
print("Best Model (based on {}):".format(best_criteria))
print(best_model.summary())
print("Selected Features:")
print(selected_columns)
