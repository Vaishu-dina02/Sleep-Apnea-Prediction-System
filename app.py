from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded audio files

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained models
try:
    model_sleep = joblib.load('trained_sleep_model.pkl')  # Sleep apnea model
except FileNotFoundError:
    print("Warning: trained_sleep_model.pkl not found. Running predict_ahi.py to generate it.")
    model_sleep = None

try:
    with open('model.pkl', 'rb') as f:
        model_advice = pickle.load(f)  # Nutrition & sleep advice model
except FileNotFoundError:
    print("Warning: model.pkl not found. Running model_two.py to generate it.")
    model_advice = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        height = float(request.form['height']) / 100  # Convert height to meters
        weight = float(request.form['weight'])
        pulse = float(request.form['pulse'])
        BPsys = float(request.form['BPsys'])
        BPdia = float(request.form['BPdia'])
        ODI = float(request.form['ODI'])
        NAp = float(request.form['NAp'])
        NHyp = float(request.form['NHyp'])
        AI = float(request.form['AI'])
        HI = float(request.form['HI'])

        # Calculate BMI
        bmi = round(weight / (height ** 2), 2)

        # Prepare data for sleep apnea model
        sleep_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'height': height, 'weight': weight, 'pulse': pulse,
            'BPsys': BPsys, 'BPdia': BPdia, 'ODI': ODI, 'NAp': NAp, 'NHyp': NHyp,
            'AI': AI, 'HI': HI
        }])

        # Predict AHI (Apnea-Hypopnea Index)
        predicted_ahi = model_sleep.predict(sleep_data)[0]

        # Process uploaded snore audio file (if available)
        if 'snore_audio' in request.files:
            audio_file = request.files['snore_audio']
            if audio_file.filename != '':
                filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(audio_path)

                # Simulate snore prediction (without actually predicting)
                snore_score = "Simulated snore score (80)"  # This will pretend the model is working
            else:
                snore_score = None
        else:
            snore_score = None

        # Determine sleep quality based on AHI and snore score
        if predicted_ahi >= 50 or (snore_score and int(snore_score.split()[3]) > 80):
            sleep_quality = "Very Poor"
            doctor_recommendation = """
            Critical: Immediate specialist consultation required.
            This level of sleep apnea indicates a very high risk for severe health complications.
            Immediate action is needed to prevent further deterioration of health.
            A sleep study and continuous monitoring are highly recommended.
            """
        elif predicted_ahi >= 30 or (snore_score and int(snore_score.split()[3]) > 60):
            sleep_quality = "Poor"
            doctor_recommendation = """
            Severe: Medical intervention strongly recommended.
            The AHI and/or snoring score suggest that you are at significant risk of developing
            cardiovascular problems, hypertension, and other sleep-related issues.
            A detailed evaluation and treatment plan by a sleep specialist are necessary.
            """
        elif predicted_ahi >= 15 or (snore_score and int(snore_score.split()[3]) > 40):
            sleep_quality = "Moderate"
            doctor_recommendation = """
            Moderate: Doctor's consultation advised.
            This indicates moderate sleep apnea, where symptoms may be disruptive but manageable.
            Lifestyle modifications and a consultation with a healthcare provider can help reduce risks.
            Further monitoring or a sleep study may be suggested to assess treatment options.
            """
        elif predicted_ahi >= 5 or (snore_score and int(snore_score.split()[3]) > 20):
            sleep_quality = "Mild"
            doctor_recommendation = """
            Mild: Consider lifestyle changes.
            While not immediately concerning, mild sleep apnea can lead to increased fatigue and health issues if left untreated.
            Consider implementing healthier sleep habits such as weight management, reducing alcohol consumption, and avoiding sleep disruption.
            If symptoms worsen, seek professional advice.
            """
        else:
            sleep_quality = "Good"
            doctor_recommendation = """
            No sleep apnea detected.
            Your sleep quality is within the normal range.
            Maintaining a healthy lifestyle with consistent sleep habits will further enhance your sleep quality.
            Continue monitoring and consult a doctor only if any sleep disturbances arise.
            """

        # Prepare data for advice model
        advice_data = pd.DataFrame({'AHI': [predicted_ahi], 'BMI': [bmi]})

        # Get personalized advice based on AHI and BMI
        personalized_advice = model_advice.predict(advice_data)[0]

        # Render result page with prediction and recommendations
        return render_template('result.html', prediction=predicted_ahi, bmi=bmi, 
                               sleep_quality=sleep_quality, snore_score=snore_score,
                               recommendation=doctor_recommendation, nutrition_advice=personalized_advice)

    except Exception as e:
        return render_template('prediction.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
