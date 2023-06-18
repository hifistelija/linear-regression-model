import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

# Assuming the data is in a dataframe df
df = pd.read_csv(r"C:\Users\ilkka\OneDrive\02)Kurssit\Python\Data Science and Machine Learning\DATA\Advertising.csv")

# Extract the predictor variables
X = df.drop("sales", axis=1)
y = df['sales']

# Create polynomial features
polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)
poly_features = polynomial_converter.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=101)

# Choosing best alpha value
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(X_train, y_train)
alpha_value = ridge_cv_model.alpha_

# Create the model
ridge_model = Ridge(alpha=alpha_value)

# Train the model
ridge_model.fit(X_train, y_train)

# Now you can use the model to make predictions on unseen data
new_data = [[230.1, 37.8, 69.2]]  # Substitute with your actual new data
# Remember to transform the new data to polynomial features
new_data_poly = polynomial_converter.transform(new_data)
sales_prediction = ridge_model.predict(new_data_poly)

print('Predicted sales:', sales_prediction)

# To check the performance of the model
y_pred = ridge_model.predict(X_test)
print('Mean Absolute error', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
