import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('nutrition.csv')

# Encode categorical columns
label_encoder = LabelEncoder()
df['Severity'] = label_encoder.fit_transform(df['Severity'])
df['Weight Category'] = label_encoder.fit_transform(df['Weight Category'])

# Encode target column 'Personalized Nutrition & Sleep Advice' as categorical label
y = df['Personalized Nutrition & Sleep Advice']  # No vectorization needed

# Prepare features (X)
X = df[['AHI', 'BMI']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Test the model (for validation)
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the trained model for later use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

def get_personalized_advice(ahi, bmi):
    # Prepare input data for prediction
    input_data = pd.DataFrame({'AHI': [ahi], 'BMI': [bmi]})

    # Predict using the trained model
    predicted_advice = model.predict(input_data)

    return predicted_advice

# Example usage (uncomment and provide values to test):
# advice = get_personalized_advice(ahi_value, bmi_value)
# print(f"Personalized Nutrition & Sleep Advice: {advice}")
