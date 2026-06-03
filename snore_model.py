import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings related to librosa
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Set the folder path
folder_path = "data/snoring"  # Adjust the path if necessary

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Load audio file

        # Extract MFCC (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)

        # Combine all features into a single array
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load data from the folder
data = []
labels = []

# Process all .wav files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing: {file_name}")  # Print each file being processed
        features = extract_features(file_path)
        
        if features is not None:
            data.append(features)
            
            # Assume the filename contains the sleep quality score (e.g., "snore_75.wav" → 75%)
            try:
                quality_score = int(file_name.split("_")[-1].replace(".wav", ""))
            except:
                quality_score = np.random.randint(50, 100)  # Assign random value if not available
                
            labels.append(quality_score)

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split into Training & Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Model
with open("snore_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as snore_model.pkl ✓")

# Cross-validation using StratifiedKFold to ensure class balance
skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)  # Reduced to 2 splits
cv_scores = []

# Split the data
for train_index, test_index in skf.split(X, y):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train_cv, y_train_cv)
    
    # Predict and calculate accuracy
    y_pred_cv = model.predict(X_test_cv)
    accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
    cv_scores.append(accuracy_cv)

# Average Cross-Validation Score
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Function to Predict Sleep Quality
def predict_sleep_quality(file_path):
    features = extract_features(file_path)
    
    if features is not None:
        features = features.reshape(1, -1)  # Reshape for prediction
        prediction = model.predict(features)[0]
        return f"Predicted Sleep Quality: {prediction}%"
    else:
        return "Error: Could not process the file."

# Example Usage
test_file = os.path.join(folder_path, "example_snore_80.wav")  # Change to a real file
if os.path.exists(test_file):
    print(predict_sleep_quality(test_file))
else:
    print(f"Example test file not found: {test_file}")
