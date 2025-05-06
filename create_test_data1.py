import numpy as np

# Load the original client data
X = np.load("client_X_0.npy")
y = np.load("client_y_0.npy")

# Optional: Shuffle before splitting (for randomness)
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# Save a subset as separate testing dataset (for example, 30% of the data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Save test data to new files
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Test dataset saved as 'X_test.npy' and 'y_test.npy'")
