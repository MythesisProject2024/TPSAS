


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("smart_city_traffic_dataset.csv")

# Extract the stream data rates
data_stream_rates = df['data_stream_rate'].values

# Prepare data for LSTM (convert to supervised learning format)
def create_dataset(data, time_window):
    X, Y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:i+time_window])
        Y.append(data[i+time_window])
    return np.array(X), np.array(Y)

time_window = 10
X, Y = create_dataset(data_stream_rates, time_window)

# Reshape for LSTM input: (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_window, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_data=(X_test, Y_test), verbose=1)

# Predict on test data
Y_pred = model.predict(X_test)

print("================ Predicted Data ==================")
print(Y_pred.flatten())  # Flatten for easier viewing

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("================ Evaluation Metrics ==================")
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared: {r2:.4f}')

# Save predicted data to a CSV file
predicted_data = pd.DataFrame({
    'Actual': Y_test.flatten(),  # Flatten to convert to 1D array
    'Predicted': Y_pred.flatten()  # Flatten to convert to 1D array
})

# Save the predicted data to a CSV file
predicted_data.to_csv('predicted_data.csv', index=False)

print("Predicted data saved to 'predicted_data.csv'")
