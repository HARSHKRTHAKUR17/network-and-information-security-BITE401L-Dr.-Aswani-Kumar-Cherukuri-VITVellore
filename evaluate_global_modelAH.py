import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load test data from client 0
X_test = np.load("client_X_0.npy")
y_test = np.load("client_y_0.npy")

# ✅ Load the global model
global_model_path = "global_model.h5"
print(f"📥 Loading Global Model: {global_model_path}")
model = load_model(global_model_path)

# 🔍 Predict class probabilities
y_pred_probs = model.predict(X_test)

# 🧠 Convert probabilities to class predictions
if y_pred_probs.shape[1] > 1:
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# 🎯 Compute metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Non-VPN", "VPN"])

# 📊 Print results
print("\n🔍 Evaluation Results:")
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📌 Confusion Matrix:")
print(conf_matrix)
print("\n📌 Classification Report:")
print(class_report)

# 📉 Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-VPN", "VPN"], yticklabels=["Non-VPN", "VPN"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
