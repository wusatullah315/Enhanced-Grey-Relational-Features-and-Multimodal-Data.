from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize variables to store results
accuracies = []

# Create K-Fold cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(X, y):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Preprocess data (fit on train, transform both)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    
    # Apply SMOTE (only on training data)
    X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_preprocessed, y_train)
    
    # Build and train model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_resampled, y_train_resampled, 
              epochs=100, batch_size=32, verbose=0)
    
    # Evaluate
    y_pred = (model.predict(X_val_preprocessed) > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    print(f"Fold accuracy: {acc:.4f}")

print(f"\nMean Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")