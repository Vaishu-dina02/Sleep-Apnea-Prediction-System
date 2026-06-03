import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your data (make sure to update this with the correct file path)
data = pd.read_csv('data/patients.csv')

# Replace commas with dots for numeric columns
numeric_columns = ['height', 'weight', 'pulse', 'ODI', 'NAp', 'NHyp', 'AI', 'HI', 'AHI']
for column in numeric_columns:
    data[column] = data[column].astype(str).str.replace(',', '.').astype(float)

# Split 'BPsys/BPdia' into separate columns
if 'BPsys/BPdia' in data.columns:
    data[['BPsys', 'BPdia']] = data['BPsys/BPdia'].str.split('/', expand=True)
    data['BPsys'] = data['BPsys'].astype(float)
    data['BPdia'] = data['BPdia'].astype(float)
else:
    print("Column 'BPsys/BPdia' not found in dataset.")

# Convert 'sex' to numeric (M -> 1, F -> 0)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])



X_train = data[['age', 'sex', 'height', 'weight', 'pulse', 'BPsys', 'BPdia', 'ODI', 'NAp', 'NHyp', 'AI', 'HI']]
y_train = data['AHI']

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_sleep_model.pkl')

# Select features and target
print(data[['height', 'weight', 'pulse', 'ODI', 'NAp', 'NHyp', 'AI', 'HI']].head())
print(data[['BPsys', 'BPdia']].head())
print(data['AHI'].head())
print(model.feature_importances_)