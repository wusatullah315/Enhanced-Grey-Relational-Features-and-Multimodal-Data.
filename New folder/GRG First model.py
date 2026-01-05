import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/kaggle/input/alzheimers-disease-dataset/alzheimers_disease_data.csv")

# Data preprocessing and feature engineering
def preprocess_data(df):
    # Drop non-informative columns
    df_cleaned = df.drop(columns=['PatientID', 'DoctorInCharge'])
    
    # Convert all boolean columns to integers (0/1)
    bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_cleaned[col] = df_cleaned[col].astype(int)
    
    # Also convert columns that might contain boolean values but have other dtype
    potential_bool_cols = ['Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                         'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
                         'MemoryComplaints', 'BehavioralProblems', 'Confusion',
                         'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                         'Forgetfulness']
    
    for col in potential_bool_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(int)
    
    # Create meaningful feature interactions
    df_cleaned['BP_Ratio'] = df_cleaned['SystolicBP'] / df_cleaned['DiastolicBP']
    df_cleaned['Cholesterol_Ratio'] = df_cleaned['CholesterolLDL'] / df_cleaned['CholesterolHDL']
    df_cleaned['MMSE_Age_Ratio'] = df_cleaned['MMSE'] / df_cleaned['Age']
    df_cleaned['Vascular_Risk'] = (df_cleaned['Hypertension'].astype(int) + 
                                 df_cleaned['CardiovascularDisease'].astype(int) + 
                                 df_cleaned['Diabetes'].astype(int))
    
    # Create composite lifestyle score
    df_cleaned['Lifestyle_Score'] = (df_cleaned['PhysicalActivity'] + 
                                   df_cleaned['DietQuality'] + 
                                   df_cleaned['SleepQuality'] - 
                                   df_cleaned['Smoking'] - 
                                   df_cleaned['AlcoholConsumption'])
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_cleaned, columns=['Gender', 'Ethnicity', 'EducationLevel'])
    
    # Drop original columns we've engineered replacements for
    df_final = df_encoded.drop(columns=['SystolicBP', 'DiastolicBP', 'CholesterolLDL', 
                                       'CholesterolHDL', 'PhysicalActivity', 
                                       'DietQuality', 'SleepQuality'])
    
    return df_final

# Apply preprocessing
df_processed = preprocess_data(df)

# Separate features and target
X = df_processed.drop(columns=['Diagnosis'])
y = df_processed['Diagnosis']

# Gray Relational Analysis implementation
def gray_relational_analysis(X, y, rho=0.5):
    """
    Perform Gray Relational Analysis between features and target
    
    Parameters:
    X - DataFrame of features
    y - Series of target values
    rho - distinguishing coefficient (default 0.5)
    
    Returns:
    Series of gray relational grades sorted in descending order
    """
    # Step 1: Normalize data
    def normalize(series):
        # Convert to numeric if not already (important fix)
        series = pd.to_numeric(series, errors='coerce')
        
        # Handle cases where all values are the same
        if series.nunique() == 1:
            return pd.Series(0.5, index=series.index)
        
        # Handle cases with NaN values
        if series.isna().any():
            series = series.fillna(series.mean())
        
        # Calculate min and max to avoid division by zero
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(1.0, index=series.index)
            
        # Perform normalization
        return (series - min_val) / (max_val - min_val)
    
    # Create a copy of X to ensure we don't modify the original
    X_copy = X.copy()
    
    # Ensure all columns are numeric before normalization
    for col in X_copy.columns:
        if X_copy[col].dtype == bool:
            X_copy[col] = X_copy[col].astype(int)
    
    # Normalize features (larger is better)
    X_normalized = X_copy.apply(normalize)
    
    # Normalize target (larger is better for reference series)
    if isinstance(y, pd.Series) and y.dtype == bool:
        y = y.astype(int)
    y_normalized = normalize(y)
    
    # Step 2: Calculate gray relational coefficients
    coefficients = pd.DataFrame(index=X.index, columns=X.columns)
    
    for col in X.columns:
        # Calculate absolute difference
        delta = np.abs(X_normalized[col] - y_normalized)
        
        # Global min and max differences
        global_min = delta.min()
        global_max = delta.max()
        
        # Handle case where all differences are the same
        if global_max == global_min:
            coefficients[col] = 1.0  # All coefficients equal when no variation
        else:
            # Calculate gray relational coefficient
            coefficients[col] = (global_min + rho * global_max) / (delta + rho * global_max)
    
    # Step 3: Calculate gray relational grades
    grades = coefficients.mean()
    
    # Step 4: Sort features by importance
    ranked_features = grades.sort_values(ascending=False)
    
    return ranked_features

# Perform GRA
gra_results = gray_relational_analysis(X, y)

# Display results
print("Gray Relational Analysis Results (Top 20 Most Influential Factors):")
print(gra_results.head(20))

# Visualization
plt.figure(figsize=(12, 8))
gra_results.head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 Factors Influencing Alzheimer\'s Diagnosis (Gray Relational Grades)')
plt.ylabel('Gray Relational Grade')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()