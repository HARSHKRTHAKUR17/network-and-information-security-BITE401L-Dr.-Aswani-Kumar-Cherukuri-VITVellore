import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Load test data
X_test = np.load("test_data.npy")
y_test = np.load("test_labels.npy")

# Load the trained global model
global_model = tf.keras.models.load_model("global_model.h5")

# Evaluate model
loss, acc = global_model.evaluate(X_test, y_test)
print(f"🧪 Global Model Accuracy: {acc:.4f}")

# Make predictions
y_pred = (global_model.predict(X_test) > 0.5).astype(int)

# Generate classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Non-VPN", "VPN"]))
