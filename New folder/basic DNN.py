from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Build a simpler model
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Dense(128, kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=0.0005, clipvalue=0.5)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

# Generate data
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)  # More data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = build_model(X_train.shape[1:])
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
    ],
    verbose=1
)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")