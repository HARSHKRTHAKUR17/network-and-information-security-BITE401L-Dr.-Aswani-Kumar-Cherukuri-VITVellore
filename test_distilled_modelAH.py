import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy", allow_pickle=True)

# 🔄 Convert string labels to 0/1
label_map = {'nonvpn': 0, 'vpn': 1}
y_test = np.array([label_map[label] for label in y_test])

# ✅ Reshape input if needed
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, axis=2)

# ✅ Load distilled global model
model = tf.keras.models.load_model("global_model_distilled.h5")

# ✅ Predict
pred_probs = model.predict(X_test)
y_pred = (pred_probs > 0.5).astype(int).flatten()

# ✅ Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-VPN", "VPN"]))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-VPN", "VPN"], yticklabels=["Non-VPN", "VPN"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix - Global Model")
plt.tight_layout()
plt.show()
