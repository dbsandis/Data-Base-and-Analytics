import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'auto_mpg_edit.csv'  # Update this path to your file location
df = pd.read_csv(file_path)

# Generate a scatterplot for acceleration against horsepower
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='horsepower', y='acceleration')
plt.title('Scatterplot of Acceleration vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Acceleration')
plt.show()
