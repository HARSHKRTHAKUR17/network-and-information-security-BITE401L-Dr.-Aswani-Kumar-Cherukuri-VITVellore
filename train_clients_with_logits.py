import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
import os

def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2)  # No activation for logits
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_clients = 5
logits_dir = "client_logits"
os.makedirs(logits_dir, exist_ok=True)

for i in range(num_clients):
    X = np.load(f"client_data_npy/client_data_{i}.npy")
    y = np.load(f"client_data_npy/client_labels_{i}.npy")
    

    model = build_model(input_shape=(X.shape[1], 1))
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X_reshaped, y, epochs=5, batch_size=32, verbose=1)

    logits = model.predict(X_reshaped)
    np.save(f"{logits_dir}/client_logits_{i}.npy", logits)
    np.save(f"{logits_dir}/client_labels_{i}.npy", y)
    print(f"✅ Client {i} logits saved")
