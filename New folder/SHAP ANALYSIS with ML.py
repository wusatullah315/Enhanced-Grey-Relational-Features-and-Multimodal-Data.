SHAP Analysis
import shap

# Explain model predictions using SHAP for XGB
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)
# Explain model predictions using SHAP for catboost
explainer = shap.Explainer(catboost)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)