# import python packages 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('house_prices.csv')

# Split the data into training and test sets
X_train = data[['square_feet', 'num_bedrooms', 'num_bathrooms', 'location']]
y_train = data['price']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
X_test = data[['square_feet', 'num_bedrooms', 'num_bathrooms', 'location']]
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = np.mean((y_pred - y_test)**2)

# Print the mean squared error
print('Mean squared error:', mse)
