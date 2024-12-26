import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load datasets
data1 = pd.read_csv('factors_dropout.csv')
data2 = pd.read_csv('score_dropout.csv')

# Step 1: Clean and Prepare `data1`
data1_clean = data1[['Marital status', 'Mother\'s qualification', 'Father\'s qualification', 
                     'Tuition fees up to date', 'Gender', 'Unemployment rate', 'GDP', 'Target']].copy()
# Ensure Target is binary and integer
data1_clean['Target'] = data1_clean['Target'].apply(lambda x: 1 if x == 'Dropout' else 0).astype(int)

# Verify unique values in Target
print("Unique values in Target (Dataset 1):", data1_clean['Target'].unique())

X1 = data1_clean.drop(columns='Target')
y1 = data1_clean['Target']


# Scale features for KNN Model 1
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

# Step 2: Clean and Prepare `data2`
data2_split = data2['access;tests;tests_grade;exam;project;project_grade;assignments;result_points;result_grade;graduate;year;acad_year'].str.split(';', expand=True)
data2_split.columns = ['access', 'tests', 'tests_grade', 'exam', 'project', 'project_grade', 
                       'assignments', 'result_points', 'result_grade', 'graduate', 'year', 'acad_year']

grade_mapping = {'A': 90, 'B': 80, 'C': 70, 'D': 60, 'F': 50}
data2_split['tests_grade'] = data2_split['tests_grade'].map(grade_mapping).fillna(0).astype(float)
data2_split['project_grade'] = data2_split['project_grade'].map(grade_mapping).fillna(0).astype(float)
data2_split['tests'] = pd.to_numeric(data2_split['tests'], errors='coerce')
data2_split['assignments'] = pd.to_numeric(data2_split['assignments'], errors='coerce')

data2_clean = data2_split[['tests', 'assignments', 'project_grade']].dropna().copy()
data2_clean['Target'] = data2_split['graduate'].astype(int)

X2 = data2_clean.drop(columns='Target')
y2 = data2_clean['Target']

# Scale features for KNN Model 2
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Step 3: Hyperparameter Tuning for KNN
param_grid = {'n_neighbors': range(1, 20)}

# KNN Model 1
grid1 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid1.fit(X1_scaled, y1)
best_knn1 = grid1.best_estimator_
print("Best KNN Model 1 Accuracy:", grid1.best_score_)

# KNN Model 2
grid2 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid2.fit(X2_scaled, y2)
best_knn2 = grid2.best_estimator_
print("Best KNN Model 2 Accuracy:", grid2.best_score_)

# Step 4: Predict on New Data
def weighted_ensemble_predict(new_data1, new_data2, model1, model2, scaler1, scaler2, weight1, weight2):
    new_data1_scaled = scaler1.transform(new_data1)
    new_data2_scaled = scaler2.transform(new_data2)
    P1 = model1.predict_proba(new_data1_scaled)[:, 1]
    P2 = model2.predict_proba(new_data2_scaled)[:, 1]
    combined_prediction = (P1 * weight1 + P2 * weight2) / (weight1 + weight2)
    return [1 if prob >= 0.5 else 0 for prob in combined_prediction]

# Input New Data
new_data1 = pd.DataFrame({'Marital status': [1], 'Mother\'s qualification': [13], 'Father\'s qualification': [2],
                          'Tuition fees up to date': [1], 'Gender': [1], 'Unemployment rate': [5.0], 'GDP': [48000]})

new_data2 = pd.DataFrame({'tests': [68.1], 'assignments': [37], 'project_grade': [90]})

# Final Prediction
final_prediction = weighted_ensemble_predict(new_data1, new_data2, best_knn1, best_knn2, scaler1, scaler2, 
                                             grid1.best_score_, grid2.best_score_)
print("Final Dropout Prediction (Weighted Ensemble):", "Dropout" if final_prediction[0] == 1 else "Graduate")
