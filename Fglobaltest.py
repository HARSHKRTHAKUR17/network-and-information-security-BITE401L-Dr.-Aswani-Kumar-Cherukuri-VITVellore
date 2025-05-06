import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = load_model("global_model_distilled.h5")

# Load test data (example from Client 0)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["nonvpn", "vpn"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["nonvpn", "vpn"], yticklabels=["nonvpn", "vpn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
