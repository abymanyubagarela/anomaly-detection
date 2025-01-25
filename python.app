from flask import Flask, request, jsonify
from joblib import load
from pyngrok import ngrok  # Menggunakan pyngrok untuk URL publik
import numpy as np

# Load model dan scaler
xgb_model = load('xgb_model.joblib')
scaler = load('scaler.joblib')

# Inisialisasi Flask app
app = Flask(__name__)

# Define endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.json

        # Extract features
        features = [
            input_data['Temperature_Motor'],
            input_data['Temperature_Gearbox'],
            input_data['Temperature_Coupling'],
            input_data['Temperature_Sprocket'],
            input_data['Temperature_Trailing'],
            input_data['Ampere_Motor'],
            input_data['Flowrate_Batubara']
        ]

        # Normalize data
        scaled_features = scaler.transform([features])

        # Predict
        prediction = xgb_model.predict(scaled_features)[0]

        # Map prediction to fault description
        fault_mapping = {
            0: "-",
            1: "Vibration",
            2: "Temperature High",
            3: "Temperature High, Overload, & Vibration",
            4: "Temperature High & Overload",
            5: "Temperature High & Vibration",
        }
        fault_description = fault_mapping.get(prediction, "Unknown Fault")

        # Return result
        return jsonify({
            'prediction': int(prediction),
            'fault_description': fault_description
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Jalankan Flask app
if __name__ == '__main__':
    # Tentukan URL ngrok
    app.run(port=5000)
