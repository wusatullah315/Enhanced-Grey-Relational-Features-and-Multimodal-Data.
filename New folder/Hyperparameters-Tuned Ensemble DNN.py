from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # Better for outliers
import numpy as np

# --- Hyperparameters ---
INIT_LR = 0.001
MIN_LR = 1e-6
BATCH_SIZE = 32
EPOCHS = 200

# --- Learning Rate Scheduler ---
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 50:
        return lr * 0.95
    else:
        return max(lr * 0.9, MIN_LR)

# --- Model Architecture ---
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Layer 1
    x = Dense(256, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Layer 2
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    optimizer = Nadam(learning_rate=INIT_LR)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

# --- Data Preparation ---
X, y = make_classification(
    n_samples=50000,  # Larger dataset
    n_features=20,
    n_classes=2,
    n_clusters_per_class=2,
    flip_y=0.001,  # Minimal label noise
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Advanced Scaling ---
scaler = RobustScaler()  # Handles outliers better
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Class Balancing ---
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}

# --- Training ---
model = build_model(X_train.shape[1:])
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, mode='max'),
        LearningRateScheduler(lr_scheduler)
    ],
    verbose=1
)

# --- Evaluation ---
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# --- Post-Training Quantization (Optional for Edge Cases) ---
# Convert to float16 for numerical stability
model.save("model.h5")
