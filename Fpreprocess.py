import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv('processed_features_labeled.csv')
df['class_label'] = LabelEncoder().fit_transform(df['class_label'])  # vpn=1, nonvpn=0

# Split by class
vpn_data = df[df['class_label'] == 1]
nonvpn_data = df[df['class_label'] == 0]

# Shuffle
vpn_data = vpn_data.sample(frac=1).reset_index(drop=True)
nonvpn_data = nonvpn_data.sample(frac=1).reset_index(drop=True)

# Define number of clients
num_clients = 3

# Split each class into clients
vpn_split = np.array_split(vpn_data, num_clients)
nonvpn_split = np.array_split(nonvpn_data, num_clients)

# Prepare and save data per client
scaler = StandardScaler()

for i in range(num_clients):
    # Combine vpn and nonvpn for client
    client_df = pd.concat([vpn_split[i], nonvpn_split[i]]).sample(frac=1).reset_index(drop=True)

    X = client_df.drop('class_label', axis=1).values
    y = client_df['class_label'].values

    # Normalize
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Save
    np.save(f'client_X_{i}.npy', X)
    np.save(f'client_y_{i}.npy', y)

    # Debug print
    unique, counts = np.unique(y, return_counts=True)
    print(f"Client {i} label distribution: {dict(zip(unique, counts))}")
