import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the User Behavior Prediction API',
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'example_input': {
                'App Usage Time (min/day)': 250,
                'Screen On Time (hours/day)': 5.2,
                'Battery Drain (mAh/day)': 1500,
                'Number of Apps Installed': 50,
                'Data Usage (MB/day)': 800,
                'Age': 28,
                'Gender': 'Male'
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
    # You might want to exit or raise an exception here depending on desired behavior
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading pickle files: {e}")
    model = None
    scaler = None

# Define the order of features as used during training
# This was inferred from the notebook: X = df.drop(columns=["User Behavior Class"])
# after 'Gender' was label encoded and other non-relevant columns were dropped.
FEATURE_ORDER = [
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age',
    'Gender'
]

@app.route('/predict', methods=['POST'])
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
        if 'Gender' not in input_df.columns:
            return jsonify({'error': 'Missing feature: Gender'}), 400
        
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x.lower() == 'male' else (0 if x.lower() == 'female' else -1))
        if (input_df['Gender'] == -1).any():
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
        # Assuming the prediction is a single class label
        predicted_class = prediction[0]

        # Convert numpy types to native Python types for JSON serialization if necessary
        if isinstance(predicted_class, np.generic):
            predicted_class = predicted_class.item()
            
        return jsonify({'predicted_user_behavior_class': predicted_class})

    except Exception as e:
        # Log the exception e for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Ensure the app runs on a port that's typically free, e.g., 5000
    # host='0.0.0.0' makes it accessible from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
