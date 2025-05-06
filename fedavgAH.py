import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Total number of clients
num_clients = 5

# Load all client models
client_models = []
for i in range(num_clients):
    model_path = f"client{i}_trained_model.h5"
    if os.path.exists(model_path):
        print(f" Loading model: {model_path}")
        model = load_model(model_path)
        client_models.append(model)
    else:
        print(f" Missing client model: {model_path}")

# Sanity check
if not client_models:
    raise ValueError(" No client models found!")

# FedAvg: Average model weights layer-by-layer
print("\n Performing Federated Averaging (FedAvg)...")
avg_model = client_models[0]  # Start from first model
avg_weights = avg_model.get_weights()

for layer_idx in range(len(avg_weights)):
    # Sum weights from all models
    layer_sum = np.sum([model.get_weights()[layer_idx] for model in client_models], axis=0)
    # Average them
    avg_weights[layer_idx] = layer_sum / len(client_models)

# Set averaged weights to base model
avg_model.set_weights(avg_weights)

# Save global model
global_model_path = "global_model.h5"
avg_model.save(global_model_path)
print(f"\n Global model saved as {global_model_path}")
