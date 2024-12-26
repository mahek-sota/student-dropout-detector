# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# # Load datasets
# data1 = pd.read_csv('factors_dropout.csv')
# data2 = pd.read_csv('score_dropout.csv')

# # Step 1: Clean and Prepare `data1` (Model 1)
# data1_clean = data1[['Marital status', 'Mother\'s qualification', 'Father\'s qualification',
#                      'Tuition fees up to date', 'Gender', 'Unemployment rate', 'GDP', 'Target']].copy()
# data1_clean['Target'] = data1_clean['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# X1 = data1_clean.drop(columns='Target')
# y1 = data1_clean['Target']

# # Scale features
# scaler1 = StandardScaler()
# X1_scaled = scaler1.fit_transform(X1)

# # Train Random Forest Model 1
# rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
# rf1.fit(X1_scaled, y1)
# rf1_accuracy = cross_val_score(rf1, X1_scaled, y1, cv=5, scoring='accuracy').mean()
# print("Random Forest Model 1 Accuracy:", rf1_accuracy)

# # Step 2: Clean and Prepare `data2` (Model 2)
# data2_split = data2['access;tests;tests_grade;exam;project;project_grade;assignments;result_points;result_grade;graduate;year;acad_year'].str.split(';', expand=True)
# data2_split.columns = ['access', 'tests', 'tests_grade', 'exam', 'project', 'project_grade',
#                        'assignments', 'result_points', 'result_grade', 'graduate', 'year', 'acad_year']

# grade_mapping = {'A': 90, 'B': 80, 'C': 70, 'D': 60, 'F': 50}
# data2_split['project_grade'] = data2_split['project_grade'].map(grade_mapping).fillna(0).astype(float)
# data2_split['tests'] = pd.to_numeric(data2_split['tests'], errors='coerce')
# data2_split['assignments'] = pd.to_numeric(data2_split['assignments'], errors='coerce')

# data2_clean = data2_split[['tests', 'assignments', 'project_grade']].dropna().copy()
# data2_clean['Target'] = data2_split['graduate'].astype(int)

# X2 = data2_clean.drop(columns='Target')
# y2 = data2_clean['Target']

# # Scale features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)

# # Train Random Forest Model 2
# rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
# rf2.fit(X2_scaled, y2)
# rf2_accuracy = cross_val_score(rf2, X2_scaled, y2, cv=5, scoring='accuracy').mean()
# print("Random Forest Model 2 Accuracy:", rf2_accuracy)

# # Step 3: Combine Predictions into a Meta-Dataset
# # Generate probabilities for both datasets
# P1_train = rf1.predict_proba(X1_scaled)[:, 1]  # Dropout probability from Model 1
# P2_train = rf2.predict_proba(X2_scaled)[:, 1]  # Dropout probability from Model 2

# # Create a new dataset with predictions as features
# meta_data = pd.DataFrame({
#     'Prediction_Model1': list(P1_train) + [0] * len(P2_train),
#     'Prediction_Model2': [0] * len(P1_train) + list(P2_train),
#     'Target': list(y1) + list(y2)
# })

# # Train a Meta-Model (Logistic Regression)
# X_meta = meta_data[['Prediction_Model1', 'Prediction_Model2']]
# y_meta = meta_data['Target']

# meta_model = LogisticRegression()
# meta_model.fit(X_meta, y_meta)

# # Step 4: Predict on New Data
# new_data1 = pd.DataFrame({'Marital status': [1], 'Mother\'s qualification': [13], 'Father\'s qualification': [2],
#                           'Tuition fees up to date': [1], 'Gender': [1], 'Unemployment rate': [5.0], 'GDP': [48000]})
# new_data1_scaled = scaler1.transform(new_data1)
# P1_new = rf1.predict_proba(new_data1_scaled)[:, 1]

# new_data2 = pd.DataFrame({'tests': [68.1], 'assignments': [37], 'project_grade': [90]})
# new_data2_scaled = scaler2.transform(new_data2)
# P2_new = rf2.predict_proba(new_data2_scaled)[:, 1]

# # Combine Predictions for Meta-Model
# meta_input = pd.DataFrame({'Prediction_Model1': P1_new, 'Prediction_Model2': P2_new})
# final_prediction = meta_model.predict(meta_input)

# # Output Final Prediction
# print("Final Dropout Prediction:", "Dropout" if final_prediction[0] == 1 else "Graduate")


# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# # Load datasets
# data1 = pd.read_csv('factors_dropout.csv')
# data2 = pd.read_csv('score_dropout.csv')

# # Step 1: Clean and Prepare Dataset 1 (factors_dropout.csv)
# data1['Target'] = data1['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# # Selecting top important features for Model 1
# X1 = data1[['Age at enrollment', 'Curricular units 1st sem (approved)',
#             'Curricular units 1st sem (evaluations)', 'Unemployment rate', 'GDP']]
# y1 = data1['Target']

# # Scale features for Model 1
# scaler1 = StandardScaler()
# X1_scaled = scaler1.fit_transform(X1)

# # Train Random Forest Model 1
# rf1 = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
# rf1.fit(X1_scaled, y1)

# # Evaluate Model 1 with Cross-Validation
# rf1_accuracy = cross_val_score(rf1, X1_scaled, y1, cv=5, scoring='accuracy').mean()
# print("Random Forest Model 1 Accuracy:", rf1_accuracy)

# # Generate probabilities for meta-model training
# P1_train = rf1.predict_proba(X1_scaled)[:, 1]  # Dropout probabilities for Dataset 1

# # Step 2: Clean and Prepare Dataset 2 (score_dropout.csv)
# # Split and extract only 'access', 'tests', 'assignments' columns
# data2_split = data2['access;tests;tests_grade;exam;project;project_grade;assignments;result_points;result_grade;graduate;year;acad_year'].str.split(';', expand=True)
# data2_split.columns = ['access', 'tests', 'tests_grade', 'exam', 'project', 'project_grade',
#                        'assignments', 'result_points', 'result_grade', 'graduate', 'year', 'acad_year']

# # Select relevant columns
# data2_clean = data2_split[['access', 'tests', 'assignments']].copy()
# data2_clean = data2_clean.dropna().astype(float)
# y2 = data2_split['graduate'].astype(int).dropna()

# # Scale features for Model 2
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(data2_clean)

# # Train Random Forest Model 2
# rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
# rf2.fit(X2_scaled, y2)

# # Evaluate Model 2 with Cross-Validation
# rf2_accuracy = cross_val_score(rf2, X2_scaled, y2, cv=5, scoring='accuracy').mean()
# print("Random Forest Model 2 Accuracy:", rf2_accuracy)

# # Generate probabilities for meta-model training
# P2_train = rf2.predict_proba(X2_scaled)[:, 1]  # Dropout probabilities for Dataset 2

# # Step 3: Combine Predictions into a Meta-Model
# # Create a meta-dataset using predictions from both models
# meta_data = pd.DataFrame({'Model1_Prob': P1_train[:len(P2_train)], 'Model2_Prob': P2_train})
# meta_target = y2[:len(meta_data)]  # Ensure target aligns with combined data length

# # Train Logistic Regression as a Meta-Model
# meta_model = LogisticRegression()
# meta_model.fit(meta_data, meta_target)

# # Evaluate Meta-Model with Accuracy
# meta_pred_train = meta_model.predict(meta_data)
# meta_accuracy = accuracy_score(meta_target, meta_pred_train)
# print("Meta-Model Accuracy:", meta_accuracy)

# # Step 4: Predict on New Data
# # New Data for Model 1
# new_data1 = pd.DataFrame({
#     'Age at enrollment': [19], 'Curricular units 1st sem (approved)': [2],
#     'Curricular units 1st sem (evaluations)': [4], 'Unemployment rate': [10.5], 'GDP': [1.74]
# })
# P1_new = rf1.predict_proba(scaler1.transform(new_data1))[:, 1]

# # New Data for Model 2
# new_data2 = pd.DataFrame({'access': [1], 'tests': [68], 'assignments': [37]})
# P2_new = rf2.predict_proba(scaler2.transform(new_data2))[:, 1]

# # Combine Predictions for Meta-Model
# meta_input = pd.DataFrame({'Model1_Prob': P1_new, 'Model2_Prob': P2_new})
# final_prediction = meta_model.predict(meta_input)

# # Final Output
# print("Final Dropout Prediction:", "Dropout" if final_prediction[0] == 1 else "Graduate")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load datasets
data1 = pd.read_csv('factors_dropout.csv')
data2 = pd.read_csv('score_dropout.csv')

# ======================= Model 1: factors_dropout.csv =======================
# Step 1: Clean and Prepare Dataset 1
data1['Target'] = data1['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Feature Engineering: Adding new features
data1['Approved_to_Enrolled'] = data1['Curricular units 1st sem (approved)'] / (
    data1['Curricular units 1st sem (enrolled)'] + 1e-5)
data1['Age_bin'] = pd.cut(data1['Age at enrollment'], bins=[15, 20, 25, 30, 35], labels=False)

# Select features
X1 = data1[['Age at enrollment', 'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (evaluations)', 'Unemployment rate', 'GDP',
            'Approved_to_Enrolled', 'Age_bin']]
y1 = data1['Target']

# Step 2: Handle Missing Values and Scale Data
imputer = SimpleImputer(strategy='mean')  # Replace NaN with column mean
X1_imputed = imputer.fit_transform(X1)    # Impute missing values

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1_imputed)

# Step 3: Handle Imbalanced Classes with SMOTE
smote = SMOTE(random_state=42)
X1_resampled, y1_resampled = smote.fit_resample(X1_scaled, y1)

# Step 4: Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X1_resampled, y1_resampled)

# Best Random Forest Model
rf1 = grid_search.best_estimator_
rf1_accuracy = cross_val_score(rf1, X1_resampled, y1_resampled, cv=5, scoring='accuracy').mean()
print("Best Random Forest Model 1 Accuracy:", rf1_accuracy)
print("Best Parameters for Model 1:", grid_search.best_params_)

# Feature Importance Plot
feature_importances = rf1.feature_importances_
feature_names = X1.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.title("Feature Importance for Model 1")
plt.show()

# XGBoost Model for Comparison
xgb1 = XGBClassifier(n_estimators=200, max_depth=20, random_state=42)
xgb1.fit(X1_resampled, y1_resampled)
xgb1_accuracy = cross_val_score(xgb1, X1_resampled, y1_resampled, cv=5, scoring='accuracy').mean()
print("XGBoost Model 1 Accuracy:", xgb1_accuracy)

# Probabilities for Meta-Model
P1_train = rf1.predict_proba(X1_scaled)[:, 1]

# ======================= Model 2: score_dropout.csv =======================
# Step 5: Clean and Prepare Dataset 2
data2_split = data2['access;tests;tests_grade;exam;project;project_grade;assignments;result_points;result_grade;graduate;year;acad_year'].str.split(';', expand=True)
data2_split.columns = ['access', 'tests', 'tests_grade', 'exam', 'project', 'project_grade',
                       'assignments', 'result_points', 'result_grade', 'graduate', 'year', 'acad_year']

# Select relevant columns
data2_clean = data2_split[['access', 'tests', 'assignments']].dropna().astype(float)
y2 = data2_split['graduate'].astype(int).dropna()

# Scale features
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(data2_clean)

# Train Random Forest Model 2
rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf2.fit(X2_scaled, y2)
rf2_accuracy = cross_val_score(rf2, X2_scaled, y2, cv=5, scoring='accuracy').mean()
print("Random Forest Model 2 Accuracy:", rf2_accuracy)

P2_train = rf2.predict_proba(X2_scaled)[:, 1]

# ======================= Meta-Model Training =======================
# Step 6: Combine Predictions into a Meta-Model
meta_data = pd.DataFrame({'Model1_Prob': P1_train[:len(P2_train)], 'Model2_Prob': P2_train})
meta_target = y2[:len(meta_data)]

# Split meta-data into train and test sets
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(meta_data, meta_target, test_size=0.3, random_state=42)

# Train Logistic Regression Meta-Model
meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y_meta_train)

# Evaluate Meta-Model
meta_pred_test = meta_model.predict(X_meta_test)
meta_accuracy = accuracy_score(y_meta_test, meta_pred_test)
print("Meta-Model Accuracy on Test Data:", meta_accuracy)

# ======================= Final Prediction =======================
# Test Case 1: Dropout (previous case)
new_data1_dropout = pd.DataFrame({
    'Age at enrollment': [19], 'Curricular units 1st sem (approved)': [2],
    'Curricular units 1st sem (evaluations)': [4], 'Unemployment rate': [10.5], 'GDP': [1.74],
    'Approved_to_Enrolled': [0.4], 'Age_bin': [1]
})
P1_dropout = rf1.predict_proba(scaler1.transform(imputer.transform(new_data1_dropout)))[:, 1]

new_data2_dropout = pd.DataFrame({'access': [1], 'tests': [68], 'assignments': [37]})
P2_dropout = rf2.predict_proba(scaler2.transform(new_data2_dropout))[:, 1]

meta_input_dropout = pd.DataFrame({'Model1_Prob': P1_dropout, 'Model2_Prob': P2_dropout})
final_prediction_dropout = meta_model.predict(meta_input_dropout)

# Test Case 2: Graduate (new test case)
new_data1_graduate = pd.DataFrame({
    'Age at enrollment': [22], 'Curricular units 1st sem (approved)': [6],
    'Curricular units 1st sem (evaluations)': [6], 'Unemployment rate': [5.0], 'GDP': [3.5],
    'Approved_to_Enrolled': [1.0], 'Age_bin': [2]
})
P1_graduate = rf1.predict_proba(scaler1.transform(imputer.transform(new_data1_graduate)))[:, 1]

new_data2_graduate = pd.DataFrame({'access': [2], 'tests': [85], 'assignments': [90]})
P2_graduate = rf2.predict_proba(scaler2.transform(new_data2_graduate))[:, 1]

meta_input_graduate = pd.DataFrame({'Model1_Prob': P1_graduate, 'Model2_Prob': P2_graduate})
final_prediction_graduate = meta_model.predict(meta_input_graduate)

# Print Predictions
print("\n=== Final Predictions ===")
print("Test Case 1 - Likely Dropout:")
print("Final Prediction:", "Dropout" if final_prediction_dropout[0] == 1 else "Graduate")

print("\nTest Case 2 - Likely Graduate:")
print("Final Prediction:", "Dropout" if final_prediction_graduate[0] == 1 else "Graduate")

