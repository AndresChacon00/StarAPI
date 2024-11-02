import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Read data from file
data = pd.read_csv("6 class csv.csv")

# Convert to numeric values
label_encoder_color = LabelEncoder()
data['Star color'] = label_encoder_color.fit_transform(data['Star color'])

label_encoder_spectral = LabelEncoder()
data['Spectral Class'] = label_encoder_spectral.fit_transform(data['Spectral Class'])

# Split evidence from labels
evidence = data.drop('Star type', axis=1)
labels = data['Star type']

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.3)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)), # Hidden layer
    tf.keras.layers.Dropout(0.4),  # Dropout layer
    tf.keras.layers.Dense(300,activation="relu"),  # Hidden layer
    tf.keras.layers.Dropout(0.4),  # Dropout layer
    tf.keras.layers.Dense(100, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(6, activation='softmax')  # Output layer
])


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