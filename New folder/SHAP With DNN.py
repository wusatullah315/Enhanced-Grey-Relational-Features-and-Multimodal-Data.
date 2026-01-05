import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model

# Step 1: Load your data
df = pd.read_csv("/kaggle/input/alzheimers-disease-dataset/alzheimers_disease_data.csv")

# Print out the columns to identify the correct target variable
print("Available columns in the dataset:")
print(df.columns.tolist())

# Step 2: Data cleaning and preprocessing
df_cleaned = df.drop(columns=['PatientID', 'DoctorInCharge'])

# Step 3: Feature engineering
df_cleaned['BP_Ratio'] = df_cleaned['SystolicBP'] / df_cleaned['DiastolicBP']
df_cleaned['Cholesterol_Ratio'] = df_cleaned['CholesterolLDL'] / df_cleaned['CholesterolHDL']
df_cleaned['MMSE_Age_Ratio'] = df_cleaned['MMSE'] / df_cleaned['Age']
df_cleaned['Vascular_Risk'] = (df_cleaned['Hypertension'].astype(int) + 
                             df_cleaned['CardiovascularDisease'].astype(int) + 
                             df_cleaned['Diabetes'].astype(int))

df_cleaned['Lifestyle_Score'] = (df_cleaned['PhysicalActivity'] + 
                               df_cleaned['DietQuality'] + 
                               df_cleaned['SleepQuality'] - 
                               df_cleaned['Smoking'] - 
                               df_cleaned['AlcoholConsumption'])

df_encoded = pd.get_dummies(df_cleaned, columns=['Gender', 'Ethnicity', 'EducationLevel'])

df_final = df_encoded.drop(columns=['SystolicBP', 'DiastolicBP', 'CholesterolLDL', 
                                   'CholesterolHDL', 'PhysicalActivity', 
                                   'DietQuality', 'SleepQuality'])

# Print columns after processing to confirm what's available
print("\nColumns after processing:")
print(df_final.columns.tolist())

# Step 4: Identify and use the correct target variable
# Looking for dementia-related target variables
dementia_columns = [col for col in df_final.columns if any(term in col.lower() for term in ['dementia', 'alzheimer', 'mmse', 'cognitive', 'diagnosis'])]
print("\nPossible target columns:")
print(dementia_columns)

# Assuming the target is 'Diagnosis' or something similar - modify based on your actual column name
# Let's try to find a suitable target column
if 'Diagnosis' in df_final.columns:
    target_column = 'Diagnosis'
elif 'DementiaStatus' in df_final.columns:
    target_column = 'DementiaStatus'
elif 'AlzheimerStatus' in df_final.columns:
    target_column = 'AlzheimerStatus'
elif dementia_columns:
    target_column = dementia_columns[0]  # Use the first relevant column found
else:
    # If nothing specific found, examine the first few rows to identify the target
    print("\nFirst 5 rows of data:")
    print(df_final.head())
    
    # Ask user to input the target column name
    target_column = input("Please enter the name of your target column from the list above: ")

print(f"\nUsing '{target_column}' as the target variable")

# Step 5: Split data into features and target
X = df_final.drop(target_column, axis=1)
y = df_final[target_column]

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Step 7: Train your model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: SHAP Analysis
feature_names = X_train.columns.tolist()
print(f"\nUsing {len(feature_names)} feature names: {feature_names[:5]}...")

# Prepare SHAP data
background_data = shap.utils.sample(X_train, min(50, len(X_train)))
test_data = X_test.iloc[:min(50, len(X_test))]

# Create prediction function for SHAP
def model_predict(x):
    preds = model.predict(x)
    if hasattr(preds, 'shape') and preds.ndim > 1 and preds.shape[1] == 1:
        return preds.flatten()
    return preds

try:
    # Try TreeExplainer first since we're using RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    
    # Summary plot (absolute impact)
    plt.figure(figsize=(12,8))
    shap.summary_plot(shap_values, test_data, 
                     feature_names=feature_names,
                     plot_type="bar",
                     show=False)
    plt.title(f"Clinical Feature Impact on {target_column}", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_feature_impact.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Beeswarm plot (detailed distribution)
    plt.figure(figsize=(12,8))
    shap.summary_plot(shap_values, test_data,
                    feature_names=feature_names,
                    plot_type="dot",
                    show=False)
    plt.title("Detailed Feature Effects", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_detailed_effects.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Individual patient explanation
    patient_idx = 0  # Change to view different patients
    
    # Create a figure for force plot
    plt.figure(figsize=(14,5))
    shap.force_plot(explainer.expected_value,
                   shap_values[patient_idx],
                   test_data.iloc[patient_idx],
                   feature_names=feature_names,
                   matplotlib=True,
                   show=False)
    
    plt.title(f"Patient Prediction Breakdown\nActual: {y_test.iloc[patient_idx]:.2f}, "
             f"Predicted: {float(model.predict(test_data.iloc[patient_idx:patient_idx+1])[0]):.2f}")
    plt.tight_layout()
    plt.savefig('patient_prediction_breakdown.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Clinical focus plots
    focus_features = ['BP_Ratio', 'MMSE_Age_Ratio', 'Vascular_Risk']
    for feature in focus_features:
        if feature in feature_names:
            plt.figure(figsize=(10,6))
            feat_idx = feature_names.index(feature)
            shap.dependence_plot(feat_idx, 
                               shap_values, 
                               test_data,
                               feature_names=feature_names,
                               show=False)
            plt.title(f"Clinical Relationship: {feature} vs {target_column}")
            plt.tight_layout()
            plt.savefig(f'clinical_relationship_{feature}.png', bbox_inches='tight', dpi=300)
            plt.show()
        else:
            print(f"Feature '{feature}' not found in the dataset. Available features: {feature_names}")
            
except Exception as e:
    print(f"\nSHAP Error: {str(e)}")
    print("\nTrying alternative SHAP method...")
    
    try:
        # Fall back to KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)
        shap_values = explainer.shap_values(test_data)
        
        # Summary plot
        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values, test_data, 
                         feature_names=feature_names,
                         plot_type="bar",
                         show=False)
        plt.title(f"Feature Importance for {target_column}")
        plt.tight_layout()
        plt.savefig('shap_feature_importance_alt.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        # Beeswarm plot
        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values, test_data,
                        feature_names=feature_names,
                        plot_type="dot",
                        show=False)
        plt.title("Feature Effects")
        plt.tight_layout()
        plt.savefig('shap_feature_effects_alt.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        # Try to show patient prediction
        plt.figure(figsize=(14, 5))
        shap.force_plot(explainer.expected_value,
                       shap_values[0],
                       test_data.iloc[0],
                       feature_names=feature_names,
                       matplotlib=True,
                       show=False)
        plt.title("Patient Prediction Breakdown")
        plt.tight_layout()
        plt.savefig('patient_prediction_alt.png', bbox_inches='tight', dpi=300)
        plt.show()
    except Exception as e2:
        print(f"Alternative SHAP method also failed: {str(e2)}")
        print("Please check if your model type is compatible with SHAP analysis.")