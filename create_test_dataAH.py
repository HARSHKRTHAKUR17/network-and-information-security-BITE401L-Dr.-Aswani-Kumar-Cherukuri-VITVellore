import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# Load preprocessed features
df = pd.read_csv("processed_features_labeled.csv")

# Drop unnecessary columns
df = df.drop(columns=["filename", "src_ip", "dst_ip"], errors='ignore')

# Map class labels to numerical values
df["class_label"] = df["class_label"].map({"nonvpn": 0, "vpn": 1})

# Separate classes
vpn_df = df[df['class_label'] == 1]
nonvpn_df = df[df['class_label'] == 0]

# Balance both classes
min_len = min(len(vpn_df), len(nonvpn_df))
vpn_df_balanced = resample(vpn_df, n_samples=min_len, random_state=123, replace=False)
nonvpn_df_balanced = resample(nonvpn_df, n_samples=min_len, random_state=123, replace=False)

# Combine and shuffle
balanced_df = pd.concat([vpn_df_balanced, nonvpn_df_balanced], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

# Split into 80% train (ignore) and 20% test
test_df = balanced_df.sample(frac=0.2, random_state=123)

# Separate features and labels
X_test = test_df.drop(columns=["class_label"]).values
y_test = test_df["class_label"].values

# Reshape X for CNN input if needed
X_test = np.expand_dims(X_test, axis=2)  # shape: (samples, features, 1)

# Save as .npy files
os.makedirs("test_data_npy", exist_ok=True)
np.save("test_data_npy/X_test.npy", X_test)
np.save("test_data_npy/y_test.npy", y_test)

print(f"✅ Balanced test data created and saved.")
print(f"🔎 Test samples: {X_test.shape[0]} | Features per sample: {X_test.shape[1]}")
