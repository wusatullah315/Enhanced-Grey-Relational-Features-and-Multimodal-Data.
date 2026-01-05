import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("/kaggle/input/alzheimers-disease-dataset/alzheimers_disease_data.csv")

# =====================================
# 1. SAFE FEATURE ENGINEERING
# =====================================
# Check which features exist before creating ratios
available_cols = set(df.columns)
feature_created = False

if 'Age' in available_cols:
    # Create ratio features only if both components exist
    if 'MMSE' in available_cols:
        df['MMSE_Age_Ratio'] = df['MMSE'] / (df['Age'] + 1e-6)
        feature_created = True
    if 'Brain_Volume' in available_cols:
        df['Brain_Age_Ratio'] = df['Brain_Volume'] / (df['Age'] + 1e-6)
        feature_created = True

# Create composite risk score if components exist
risk_factors = ['Hypertension', 'CardiovascularDisease', 'Diabetes']
existing_risk_factors = [col for col in risk_factors if col in available_cols]

if len(existing_risk_factors) > 0:
    df['Vascular_Risk'] = df[existing_risk_factors].sum(axis=1)
    feature_created = True

if not feature_created:
    print("Note: No additional features were created - using original features")

# =====================================
# 2. DATA PREPROCESSING
# =====================================
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Automatic column type detection
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# =====================================
# 3. OPTIMIZED MODEL TRAINING
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.15, random_state=42, stratify=y
)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train XGBoost with automatic parameter optimization
xgb = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=len(y_res[y_res==0])/len(y_res[y_res==1]),
    eval_metric='auc',
    use_label_encoder=False,
    early_stopping_rounds=50
)

xgb.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=10)

# =====================================
# 4. EVALUATION
# =====================================
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:,1]

print("\n=== Optimized Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Show top 10 most important features
try:
    feature_names = preprocessor.get_feature_names_out()
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print("\nTop 10 Features:")
    print(feat_imp)
except AttributeError:
    print("\nFeature importance not available with current scikit-learn version")