import pickle
import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')#home route endpoint
def home():
    return jsonify({
        'message': 'Welcome to the User Behavior Prediction API',
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'example_input': {
                'appUsageTime': 250,
                'screenOnTime': 5.2,
                'batteryDrain': 1500,
                'appsInstalled': 50,
                'dataUsage': 800,
                'age': 28,
                'gender': 'Male'
            }
        }
    })

# Load the model and scaler
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print(f"Error: Ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the same directory as app.py.")
    
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading pickle files: {e}")#Catch any other exceptions that may occur during loading
    model = None
    scaler = None

# Define the order of features as used during training
FEATURE_ORDER = [
    'appUsageTime',
    'screenOnTime',
    'batteryDrain',
    'appsInstalled',
    'dataUsage',
    'age',
    'gender'
]

@app.route('/predict', methods=['POST'])#prediction endpoint
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON input'}), 400

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Preprocessing
        # 1. Gender Encoding: Assuming 'Male' -> 1, 'Female' -> 0
        # LabelEncoder in scikit-learn assigns alphabetical order, so Female (0), Male (1)
        if 'gender' not in input_df.columns:
            return jsonify({'error': 'Missing feature: gender'}), 400
        
        input_df['gender'] = input_df['gender'].apply(lambda x: 1 if x.lower() == 'male' else (0 if x.lower() == 'female' else -1))
        if (input_df['gender'] == -1).any():
            return jsonify({'error': 'Invalid Gender value. Must be "Male" or "Female".'}), 400

        # 2. Ensure all features are present and in the correct order
        try:
            processed_df = input_df[FEATURE_ORDER]
        except KeyError as e:
            return jsonify({'error': f'Missing feature: {str(e)}'}), 400
        
        # 3. Scaling
        # The scaler expects a NumPy array
        scaled_features = scaler.transform(processed_df.values)
        
        # Prediction
        prediction = model.predict(scaled_features)
        
        # The model.predict might return a numpy array, get the first element
        #predicted class is of type numpy.int64
        predicted_class = prediction[0]
        print(f"Predicted class type: {type(predicted_class)}")  

        # Convert numpy types to native Python types for JSON as Flask jsonify does not handle numpy types well
        if isinstance(predicted_class, np.generic):
            predicted_class = predicted_class.item()
            
        return jsonify({'predicted_user_behavior_class': predicted_class})

    except Exception as e:
        # Log the exception e for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':#when the script is run directly, name is assigned to __main__
    #use 5000 as default port if PORT environment variable is not set
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
