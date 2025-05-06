import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# Load preprocessed features
df = pd.read_csv("processed_features_labeled.csv")

# Drop unnecessary columns (keep only numerical features)
df = df.drop(columns=["filename", "src_ip", "dst_ip"], errors='ignore')

# Map class labels to numerical values
df["class_label"] = df["class_label"].map({"nonvpn": 0, "vpn": 1})

# Separate into two classes
vpn_df = df[df['class_label'] == 1]
nonvpn_df = df[df['class_label'] == 0]

# Balance both classes
min_len = min(len(vpn_df), len(nonvpn_df))
vpn_df = resample(vpn_df, n_samples=min_len, random_state=42, replace=False)
nonvpn_df = resample(nonvpn_df, n_samples=min_len, random_state=42, replace=False)

# Merge and shuffle
balanced_df = pd.concat([vpn_df, nonvpn_df], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = balanced_df.drop(columns=["class_label"]).values
y = balanced_df["class_label"].values

# Split into 5 clients
num_clients = 5
X_splits = np.array_split(X, num_clients)
y_splits = np.array_split(y, num_clients)

# Create directory to store .npy files
os.makedirs("client_data_npy", exist_ok=True)

# Save each client's data and labels
for i in range(num_clients):
    np.save(f"client_data_npy/client_data_{i}.npy", X_splits[i])
    np.save(f"client_data_npy/client_labels_{i}.npy", y_splits[i])
    print(f"✅ Saved Client {i} with {len(X_splits[i])} samples")

print("\n🎯 Balanced client datasets saved successfully!")
