import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, 
                                   Dropout, LeakyReLU, GaussianNoise)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint)
from imblearn.over_sampling import SMOTE

# 1. Data Preparation (assuming X_train, X_test, y_train, y_test exist)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# 2. Model Architecture
def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Input regularization
    x = GaussianNoise(0.05)(inputs)
    
    # Feature extraction block
    x = Dense(128, kernel_regularizer=l2(0.02))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Feature processing block
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Decision block
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.05)(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

model = build_advanced_model(X_resampled.shape[1:])
model.summary()

# 3. Training Configuration
callbacks = [
    EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc')  # Changed to .keras
]

history = model.fit(
    X_resampled, y_resampled,
    validation_split=0.15,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# 4. Evaluation Functions
def evaluate_model(model, X_test, y_test):
    # Basic metrics
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'auc': results[2],
        'precision': results[3],
        'recall': results[4]
    }
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    
    # Find optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Best F1-Score: {f1_scores[optimal_idx]:.4f}")
    
    metrics['optimal_threshold'] = optimal_threshold
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    return metrics

# Run evaluation
metrics = evaluate_model(model, X_test_scaled, y_test)

# 5. Final Performance Summary
print("\nFinal Model Performance:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")

# 6. Training History Visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('AUC over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision over Epochs')
plt.legend()

plt.tight_layout()
plt.show()