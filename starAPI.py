from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
app = Flask(__name__)

# Load modelo
model = tf.keras.models.load_model("classificatorModel.h5")

# Define API's route
@app.route('/predict', methods=['POST'])
def predict():
    # Check is file exists
    if 'file' not in request.files:
        return jsonify({'error': 'No file was send'})
    
    file = request.files['file']

    # Check file's name
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read CSV
    data = pd.read_csv(file)

    # Asegurarse de que las columnas coincidan con las del entrenamiento
    expected_columns = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
    if list(data.columns) != expected_columns:
        return jsonify({'error': 'Invalid columns in CSV file'})
    
    # Preprocess data
    scaler = joblib.load('scaler.pkl')
    data_scaled = scaler.transform(data)

    # Predict
    predictions = model.predict(data_scaled)

    # Convert to list
    predictions_list = np.argmax(predictions, axis=1).tolist()


    # Return as JSON
    return jsonify({'predictions': predictions_list})

if __name__ == '__main__':
    app.run(debug=True)