import tensorflow as tf
import numpy as np

# Define CNN model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),  
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the client datasets
num_clients = 5
client_data = []
client_labels = []

for i in range(num_clients):
    X_client = np.load(f"client_data_{i}.npy")
    y_client = np.load(f"client_labels_{i}.npy")
    
    # Reshape for CNN input
    X_client = X_client.reshape((X_client.shape[0], X_client.shape[1], 1))  
    client_data.append(X_client)
    client_labels.append(y_client)

# Get input shape dynamically
input_shape = (client_data[0].shape[1], 1)

# Initialize the global model with correct input shape
global_model = create_model(input_shape)

# Train the model iteratively on each client's data
for round_num in range(3):  # Simulating 3 training rounds
    print(f"\n---- Federated Round {round_num + 1} ----")
    
    for client_id in range(num_clients):
        print(f"Training on Client {client_id + 1}")
        global_model.fit(client_data[client_id], client_labels[client_id], epochs=2, batch_size=32, verbose=1)

print("Federated-like training completed!")

# Save the final trained global model
global_model.save("global_federated_model.h5")
