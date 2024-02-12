import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming df is loaded from 'auto_mpg_edit.csv'
df = pd.read_csv('auto_mpg_edit.csv')

# Example selected_columns based on AIC criteria, adjust accordingly
selected_columns = ['weight', 'horsepower', 'cylinders']  # Example features

# Add constant for intercept in model training and predictions
X = sm.add_constant(df[selected_columns])
y = df['lmpg']  # Assuming 'lmpg' is the log-transformed 'mpg'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = sm.OLS(y_train, X_train).fit()

# Predict using the model; ensure X_test also includes a constant if the model expects one
y_pred = model.predict(sm.add_constant(X_test[selected_columns]))  # Corrected line

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
