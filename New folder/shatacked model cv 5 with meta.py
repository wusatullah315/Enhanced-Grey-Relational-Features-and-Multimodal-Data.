# First define all your components
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Define your preprocessor (example)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# 2. Define base models
models = {
    'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'lgbm': LGBMClassifier(random_state=42),
    'cat': CatBoostClassifier(verbose=0, random_state=42),
    'rf': RandomForestClassifier(random_state=42)
}
# Create meta-features
meta_features = []
for name, model in models.items():
    pipe = make_pipeline(preprocessor, model)  # Now properly imported
    meta_feature = cross_val_predict(pipe, X_train, y_train, 
                                   cv=5, method='predict_proba')[:,1]
    meta_features.append(meta_feature)
    
X_meta = np.column_stack(meta_features)

# Train meta-model
meta_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0)
meta_model.fit(X_meta, y_train)

# Evaluate
test_meta_features = []
for name, model in models.items():
    pipe = make_pipeline(preprocessor, model)
    pipe.fit(X_train, y_train)
    test_meta_feature = pipe.predict_proba(X_test)[:,1]
    test_meta_features.append(test_meta_feature)
    
X_test_meta = np.column_stack(test_meta_features)
y_pred_meta = meta_model.predict(X_test_meta)

print("Stacked Model Accuracy:", accuracy_score(y_test, y_pred_meta))