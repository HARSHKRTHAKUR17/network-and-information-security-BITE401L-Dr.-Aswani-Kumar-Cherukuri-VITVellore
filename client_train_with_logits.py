# Fclient_train.py
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

# Load your client data
client_id = 2 # change this to 0, 1, 2 per client run
X = np.load(f'client_X_{client_id}.npy')
y = np.load(f'client_y_{client_id}.npy')
y_cat = to_categorical(y, num_classes=2)

# Train model
model = create_model(X.shape[1:])
model.fit(X, y_cat, epochs=3, batch_size=256, verbose=1)

# Save logits (soft labels)
logits = model.predict(X)
np.save(f'client_logits_{client_id}.npy', logits)
np.save(f'client_features_{client_id}.npy', X)
np.save(f'client_labels_{client_id}.npy', y_cat)

# Save model if needed
model.save(f'client_model_{client_id}.h5')
