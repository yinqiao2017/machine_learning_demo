import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split


# Function to convert categorical features to ordered numeric codes
def convert_to_numeric(df, column, order):
    df[column] = pd.Categorical(df[column], categories=order, ordered=True)
    df[column] = df[column].cat.codes
    return df

# Generating synthetic data for demonstration
# Load the CSV file
data = pd.read_csv('Predict Student Dropout and Academic Success.csv', delimiter=';')
df = pd.DataFrame(data)

# Features to Label Encode
label_features = {
  'Target' : ['Dropout','Graduate'],
}

# Apply the function to each specified column
for col, order in label_features.items():
    df = convert_to_numeric(df, col, order)

# Getting x_data (features) and y_data (target)
x_data = df.iloc[:, :-1]  # Select all rows and all columns except the last one
y_data = df.iloc[:, -1]    # Select all rows and only the last column

# Split the data (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', penalty='l1', C=0.5)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_val, y_pred)
print('Classification Report:')
print(class_report)



