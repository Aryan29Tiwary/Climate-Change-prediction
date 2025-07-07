# Climate-Change-prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("climate_change_dataset.csv")

# Identify columns with 'object' dtype (likely containing strings)
object_columns = data.select_dtypes(include=['object']).columns

# Create a LabelEncoder for each object column and transform the data
for col in object_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Encode string values to numerical labels

# Impute missing values using the mean for features
imputer = SimpleImputer(strategy='mean')
X = data.drop(columns=['Urbanization_Index'])  # Features
X = imputer.fit_transform(X)  # Fit and transform the imputer on your features

# Extract target variable and drop rows with missing target values
y = data['Urbanization_Index']
y = y.dropna()  # Drop rows where target is NaN
X = X[y.index]  # Align features with target

# Apply PCA
pca = PCA(n_components=min(15, X.shape[1]))  # Ensure n_components does not exceed number of features
X_pca = pca.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Urbanization Index")
plt.ylabel("Predicted Urbanization Index")
plt.title("Actual vs. Predicted Urbanization Index")
plt.grid(True)
plt.show()

# User input for prediction
user_input = []
for feature in data.drop(columns=['Urbanization_Index']).columns:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            user_input.append(value)
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert user input to a DataFrame and apply PCA
user_input = np.array(user_input).reshape(1, -1)  # Reshape for a single sample
user_input_pca = pca.transform(user_input)  # Transform using PCA

# Make prediction based on user input
user_prediction = model.predict(user_input_pca)

print(f"Predicted Urbanization Index for the provided input: {user_prediction[0]}")![image](https://github.com/user-attachments/assets/9d63cf0d-3455-4f8c-be6d-c76798428e36)
