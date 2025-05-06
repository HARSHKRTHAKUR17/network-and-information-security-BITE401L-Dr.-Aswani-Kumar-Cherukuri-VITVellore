import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.utils import class_weight
import os

# ✅ Load all client logits and true labels
all_logits = []
all_labels = []

for client_id in range(5):
    soft_path = f"client_soft_logits/client{client_id}_soft_logits.npy"
    label_path = f"client_soft_logits/client{client_id}_true_labels.npy"

    if os.path.exists(soft_path) and os.path.exists(label_path):
        all_logits.append(np.load(soft_path))
        all_labels.append(np.load(label_path))

# ✅ Concatenate logits and labels from all clients
X_logits = np.vstack(all_logits)
y_true = np.hstack(all_labels)

# ✅ Normalize logits using softmax (temperature T = 2.0)
temperature = 2.0
soft_labels = tf.nn.softmax(X_logits / temperature).numpy()

# ✅ Create a new model (same architecture as base model)
def create_model():
    inputs = Input(shape=(4, 1))
    x = Conv1D(32, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

model = create_model()

# ✅ Compile with combined loss: distillation + hard labels
def custom_loss(y_true, y_pred):
    # Hard loss (true label)
    hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Soft label loss
    soft_target = tf.convert_to_tensor(soft_labels, dtype=tf.float32)
    pred_logits = tf.math.log(tf.clip_by_value(y_pred, 1e-9, 1.0))  # log predictions
    soft_loss = tf.keras.losses.KLD(soft_target, pred_logits)

    return 0.5 * hard_loss + 0.5 * soft_loss

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# ✅ Add class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_true),
    y=y_true
)
cw_dict = dict(enumerate(class_weights))

# ✅ Reshape for input
X_input = np.expand_dims(X_logits[:, :4], axis=-1)

# ✅ Train
print("🎯 Training global model via improved distillation...")
model.fit(
    X_input,
    y_true,
    epochs=5,
    batch_size=64,
    class_weight=cw_dict,
    verbose=2
)

# ✅ Save final model
model.save("global_model_distilled1.h5")
print("✅ Global model saved as: global_model_distilled.h5")
