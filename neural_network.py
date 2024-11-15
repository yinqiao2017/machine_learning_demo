import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

# Create the MLPClassifier model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Load the CSV file
data = pd.read_csv('StudentPerformanceFactors.csv')

df = pd.DataFrame(data)

# Features to Label Encode
label_features = {
  'Parental_Involvement' : ['Low','Medium','High'],
  'Access_to_Resources' : ['Low','Medium','High'],
  'Motivation_Level' : ['Low','Medium','High'],
  'Family_Income' : ['Low','Medium','High'],
  'Teacher_Quality' : ['Low','Medium','High'],
  'Peer_Influence' : ['Negative','Neutral','Positive'],
  'Distance_from_Home' : ['Near','Moderate','Far'],
  'Parental_Education_Level' : ['High School','College','Postgraduate'],
  'Extracurricular_Activities' : ['No','Yes'],
  'Internet_Access' : ['No','Yes'],
  'School_Type' : ['Public','Private'],
  'Learning_Disabilities' : ['Yes','No'],
  'Gender' : ['Male','Female'],
}

# Apply the function to each specified column
for col, order in label_features.items():
    df = convert_to_numeric(df, col, order)

# Getting x_data (features) and y_data (target)
x_data = df.iloc[:, :-1]  # Select all rows and all columns except the last one
y_data = df.iloc[:, -1]    # Select all rows and only the last column

# Split the data (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_reg.fit(X_train, y_train)
y_predict = mlp_reg.predict(X_val)

# Evaluate the model
# Assume y_true is the actual values and y_pred are the predicted values
mse = mean_squared_error(y_val, y_predict)
mae = mean_absolute_error(y_val, y_predict)
r2 = r2_score(y_val, y_predict)

# Printing the metrics
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

