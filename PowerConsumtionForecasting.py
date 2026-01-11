# =====================================================
# SECTION 1: Import Packages
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# SECTION 2: Load Dataset
# =====================================================
df = pd.read_csv("DataSet/powerconsumption.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.sort_values('Datetime', inplace=True)
df.set_index('Datetime', inplace=True)
df.head()

# =====================================================
# SECTION 3: Data Visualization
# =====================================================

# Pairplot for main features
sns.pairplot(df[['Temperature', 'Humidity', 'WindSpeed',
                 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']])
plt.show()

# Time series plot for each zone
plt.figure(figsize=(12,6))
sns.lineplot(data=df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']])
plt.xlabel("Datetime")
plt.ylabel("Power Consumption")
plt.title("Power Consumption Time Series")
plt.legend(['Zone 1', 'Zone 2', 'Zone 3'])
plt.show()

# =====================================================
# SECTION 4: Feature Scaling
# =====================================================
features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows']
targets = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[targets])

# =====================================================
# SECTION 5: Create Time Series Sequences
# =====================================================
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 24  # Using last 24 timesteps (~4 hours if 10-min interval)
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Split into train/test
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# =====================================================
# SECTION 6: LSTM Model
# =====================================================
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, len(features))),
    Dense(3)  # One output per zone
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.summary()

# Train LSTM
lstm_history = model_lstm.fit(X_train, y_train,
                              validation_data=(X_test, y_test),
                              epochs=20,
                              batch_size=64,
                              verbose=2)

# =====================================================
# SECTION 7: GRU Model
# =====================================================
model_gru = Sequential([
    GRU(50, activation='relu', input_shape=(SEQ_LENGTH, len(features))),
    Dense(3)
])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.summary()

# Train GRU
gru_history = model_gru.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=20,
                            batch_size=64,
                            verbose=2)

# =====================================================
# SECTION 8: Model Evaluation
# =====================================================
# Predictions
y_pred_lstm = model_lstm.predict(X_test)
y_pred_gru = model_gru.predict(X_test)

# Inverse scaling
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
y_pred_gru_inv = scaler_y.inverse_transform(y_pred_gru)

# Compute Metrics
for i, zone in enumerate(targets):
    mse_lstm = mean_squared_error(y_test_inv[:,i], y_pred_lstm_inv[:,i])
    mae_lstm = mean_absolute_error(y_test_inv[:,i], y_pred_lstm_inv[:,i])
    mse_gru = mean_squared_error(y_test_inv[:,i], y_pred_gru_inv[:,i])
    mae_gru = mean_absolute_error(y_test_inv[:,i], y_pred_gru_inv[:,i])
    
    print(f"{zone} - LSTM: MSE={mse_lstm:.2f}, MAE={mae_lstm:.2f} | GRU: MSE={mse_gru:.2f}, MAE={mae_gru:.2f}")

# =====================================================
# SECTION 9: Plot Forecast vs Real
# =====================================================
plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:200,0], label='Actual Zone 1')
plt.plot(y_pred_lstm_inv[:200,0], label='LSTM Predicted Zone 1')
plt.plot(y_pred_gru_inv[:200,0], label='GRU Predicted Zone 1')
plt.title('Zone 1 Forecast Comparison (First 200 samples)')
plt.xlabel('Time step')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:200,1], label='Actual Zone 2')
plt.plot(y_pred_lstm_inv[:200,1], label='LSTM Predicted Zone 2')
plt.plot(y_pred_gru_inv[:200,1], label='GRU Predicted Zone 2')
plt.title('Zone 2 Forecast Comparison (First 200 samples)')
plt.xlabel('Time step')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:200,2], label='Actual Zone 3')
plt.plot(y_pred_lstm_inv[:200,2], label='LSTM Predicted Zone 3')
plt.plot(y_pred_gru_inv[:200,2], label='GRU Predicted Zone 3')
plt.title('Zone 3 Forecast Comparison (First 200 samples)')
plt.xlabel('Time step')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

# =====================================================
# SECTION 10: Plot Training Loss (Dashboard-like)
# =====================================================
plt.figure(figsize=(12,5))
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
plt.plot(gru_history.history['loss'], label='GRU Train Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Val Loss')
plt.title('Training and Validation Loss for LSTM and GRU')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
