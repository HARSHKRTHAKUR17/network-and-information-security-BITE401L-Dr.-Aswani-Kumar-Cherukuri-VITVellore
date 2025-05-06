import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Change client ID (0,1,2,3,4)
client_id = 4

# Load client data
X_train = np.load(f"client_data_{client_id}.npy")
y_train = np.load(f"client_labels_{client_id}.npy")

# Define a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (VPN vs Non-VPN)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train locally
print(f"🚀 Training Client {client_id} Model...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the local model
model.save(f"client_model_{client_id}.h5")
print(f"✅ Client {client_id} Model Saved!")
