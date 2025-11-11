import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv("data/candidate_data.csv")
X = data[['Experience', 'Education', 'Skills', 'Interview_Score']].values
y = data['Eligible'].values
X = X / X.max(axis=0)  # Normalize data

# Add regularization to prevent overfitting
model = Sequential([
    Dense(12, activation='relu', input_shape=(4,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Use class weights if data is imbalanced
class_weights = {0: 1.5, 1: 1.0}  # Give more weight to ineligible cases if needed

# Train with more focus on precision
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train with validation split to monitor performance
model.fit(X, y, epochs=150, batch_size=32, validation_split=0.2, 
          class_weight=class_weights, callbacks=[early_stopping])