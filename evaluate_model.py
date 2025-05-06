import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the predictions data
df = pd.read_csv("predictions.csv")  # Load the data where predictions are saved

# Check and display the first few rows of the dataframe to confirm column names
print(df.head())

# Convert 'actual_label' to numeric if it's a string
if df['class_label'].dtype == 'O':  # Check if it's a string type
    df['class_label'] = df['class_label'].map({'vpn': 1, 'nonvpn': 0})  # Map 'vpn' to 1 and 'nonvpn' to 0

# Extract true and predicted labels
y_true = df['class_label'].values
y_pred = df['predicted_label'].values

# Compute confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

cr = classification_report(y_true, y_pred)
print("Classification Report:")
print(cr)
