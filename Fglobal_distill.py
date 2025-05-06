# Fglobal_distill.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load client logits, features, and labels
client_logits = []
client_features = []
client_labels = []

for i in range(3):
    client_logits.append(np.load(f'client_logits_{i}.npy'))
    client_features.append(np.load(f'client_features_{i}.npy'))
    client_labels.append(np.load(f'client_labels_{i}.npy'))

# Combine from all clients
X_all = np.concatenate(client_features, axis=0)
logits_all = np.concatenate(client_logits, axis=0)
true_labels = np.concatenate(client_labels, axis=0)

# Distill using client logits as soft targets
global_model = create_model(X_all.shape[1:])
global_model.fit(X_all, logits_all, epochs=5, batch_size=512, verbose=1)

# Save final model
global_model.save("global_model_distilled.h5")
