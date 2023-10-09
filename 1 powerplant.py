
# 1.) Build a linear regression model on the given power plant dataset. Create the
# training and testing set. Make predictions of the datapoints in the test set.
# Evaluate the model using appropriate performance matrix


# pip install pandas scikit-learn numpy
# pip install openpyxl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_excel('Data\Power Plant.xlsx')

# Split the dataset into features (X) and target variable (y)
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
