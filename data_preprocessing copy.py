import pandas as pd
import numpy as np

# Load the processed data
df = pd.read_csv("processed_features_labeled.csv")

# Split the data into features (X) and labels (y)
X = df.drop(columns=['class_label']).values  # Features
y = df['class_label'].values  # Target label (VPN or Non-VPN)

# Convert labels to numerical format if needed (e.g., 'vpn' -> 1, 'nonvpn' -> 0)
label_mapping = {'vpn': 1, 'nonvpn': 0}
y = np.array([label_mapping[label] for label in y])

# Split the data into "clients" (e.g., 5 clients)
num_clients = 5
X_splits = np.array_split(X, num_clients)
y_splits = np.array_split(y, num_clients)

# Save each client's data separately
for i in range(num_clients):
    np.save(f"client_data_{i}.npy", np.array(X_splits[i]))  # Save features
    np.save(f"client_labels_{i}.npy", np.array(y_splits[i]))  # Save labels

print("Client data successfully saved!")
