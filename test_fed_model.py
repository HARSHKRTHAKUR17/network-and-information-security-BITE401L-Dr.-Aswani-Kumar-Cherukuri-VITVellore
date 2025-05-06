import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load test dataset
df_test = pd.read_csv("processed_features_labeled.csv")  # Ensure this file has correct data

# Check if dataset contains both VPN and Non-VPN labels
df_vpn = df_test[df_test['class_label'] == 'vpn']
df_nonvpn = df_test[df_test['class_label'] == 'nonvpn']

if df_vpn.empty or df_nonvpn.empty:
    raise ValueError("⚠️ Test dataset must contain both VPN and Non-VPN samples!")

# Balance dataset by taking equal samples
min_samples = min(len(df_vpn), len(df_nonvpn))
df_test_balanced = pd.concat([df_vpn.sample(min_samples), df_nonvpn.sample(min_samples)])

# Reduce dataset size for faster testing
df_test_balanced = df_test_balanced.sample(n=10000, random_state=42)  # Use 10k samples

# Shuffle dataset
df_test_balanced = df_test_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Print class distribution
print("\nClass Distribution in Test Data:")
print(df_test_balanced['class_label'].value_counts())

# Prepare features and labels
X_test = df_test_balanced.drop(columns=['class_label']).values
y_test = df_test_balanced['class_label'].map({'vpn': 1, 'nonvpn': 0}).values

# Reshape input for CNN
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Load trained model
model = tf.keras.models.load_model("global_federated_model.h5")

# Predict
y_pred = model.predict(X_test)

# Ensure predictions select the correct class
y_pred = np.argmax(y_pred, axis=1)

# Print accuracy
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=['Non-VPN', 'VPN']))
