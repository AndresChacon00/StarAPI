import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
# Read data from file
data = pd.read_csv("6 class csv.csv")

# Split evidence from labels
evidence = data.drop(['Star type', 'Star color', 'Spectral Class'], axis=1)
labels = data['Star type']

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.3)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# MODEL
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(4,)))
model.add(tf.keras.layers.Dense(300,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(200,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(6,activation = "softmax"))

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate how well model performs
model.evaluate(X_test, y_test, verbose=2)


# Save model
model.save('classificatorModel.h5')

manual_input = np.array([[3042, 0.0005, 0.1542, 16.6],
                         [2600, 0.0003, 0.102, 18.7],
                         [2800, 0.0002, 0.16, 16.65],
                         [1939, 0.000138, 0.103, 20.06],
                         [3600, 0.0029, 0.51, 10.69]])

# Normalize the manual input using the same scaler
manual_input_scaled = scaler.transform(manual_input)

# Predict
manual_predictions = model.predict(manual_input_scaled)

# Convert predictions to class labels
manual_predicted_labels = np.argmax(manual_predictions, axis=1)

# Print the predictions
print(manual_predicted_labels)