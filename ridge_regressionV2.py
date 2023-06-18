import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("sales", axis=1)
    y = df['sales']
    polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)
    return poly_features, y, polynomial_converter


def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.33, random_state=101)


def train_model(X_train, y_train):
    ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
    ridge_cv_model.fit(X_train, y_train)
    ridge_model = Ridge(alpha=ridge_cv_model.alpha_)
    ridge_model.fit(X_train, y_train)
    return ridge_model


def predict_and_evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    return mae, rmse, r2


def predict_new_data(model, new_data, polynomial_converter):
    new_data_poly = polynomial_converter.transform(new_data)
    sales_prediction = model.predict(new_data_poly)
    return sales_prediction

# Usage:
file_path = r"C:\Users\ilkka\OneDrive\02)Kurssit\Python\Data Science and Machine Learning\DATA\Advertising.csv"
X, y, polynomial_converter = load_and_preprocess_data(file_path)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)
model = train_model(X_train, y_train)
mae, rmse, r2 = predict_and_evaluate_model(model, X_test, y_test)
new_data = [[230.1, 37.8, 69.2]]  # Substitute with your actual new data
sales_prediction = predict_new_data(model, new_data, polynomial_converter)

print('Predicted sales:', sales_prediction)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
