import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the processed dataset
# Ensure 'class_label' column contains 'vpn' and 'nonvpn'
df = pd.read_csv("processed_features_labeled.csv")

# Convert labels to numerical format
label_mapping = {'vpn': 1, 'nonvpn': 0}
df['class_label'] = df['class_label'].map(label_mapping)

# Separate VPN and Non-VPN samples
vpn_data = df[df['class_label'] == 1]
nonvpn_data = df[df['class_label'] == 0]

# Determine the minimum class count to create a balanced test set
min_samples = min(len(vpn_data), len(nonvpn_data))

# Take an equal number of VPN and Non-VPN samples for the test set
vpn_test, _ = train_test_split(vpn_data, test_size=0.7, random_state=42)  # 30% for test
nonvpn_test, _ = train_test_split(nonvpn_data, test_size=0.7, random_state=42)

# Combine and shuffle the test set
test_data = pd.concat([vpn_test, nonvpn_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
X_test = test_data.drop(columns=['class_label']).values
y_test = test_data['class_label'].values

# Save the test dataset
np.save("test_data.npy", X_test)
np.save("test_labels.npy", y_test)

print(f"✅ Test dataset created successfully! Total samples: {len(X_test)}")
