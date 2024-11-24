from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.load_model("classificatorModel.h5")

# Cargar el scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return 'Hello, World!'

# Definir la ruta para la API
@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    
    # Verificar si se ha enviado un archivo
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Verificar si el archivo tiene un nombre
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})
    
    # Leer el archivo CSV
    data = pd.read_csv(file)
    print("CSV data read successfully")
    
    # Asegurarse de que las columnas coincidan con las del entrenamiento
    expected_columns = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
    if list(data.columns) != expected_columns:
        print("Invalid columns in CSV file")
        return jsonify({'error': 'Invalid columns in CSV file'})
    
    # Preprocesar los datos
    data_scaled = scaler.transform(data)
    print("Data scaled successfully")
    
    # Hacer predicciones
    predictions = model.predict(data_scaled)
    print("Predictions made successfully")
    
    # Convertir las predicciones a una lista
    predictions_list = np.argmax(predictions, axis=1).tolist()
    print("Predictions converted to list")
    
    # Devolver las predicciones como un JSON
    return jsonify({'predictions': predictions_list})

if __name__ == '__main__':
    app.run(debug=True)