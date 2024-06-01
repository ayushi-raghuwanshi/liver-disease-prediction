from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load('rf_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names used during training
feature_names = ['GGT', 'AAP', 'TB', 'DB', 'IB', 'DELTA B']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json

    # Convert JSON to DataFrame
    input_data = pd.DataFrame([data])

    # Ensure the columns are in the correct order
    input_data = input_data[feature_names]

    # Normalize the input data using the loaded scaler
    input_data = scaler.transform(input_data)

    print(input_data);
    # Get predictions
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    # Create response
    response = {
        'prediction': int(prediction[0]),       # Convert prediction to int
        'probability': float(probability[0])    # Convert probability to float
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
