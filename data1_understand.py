import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load Dataset
data1 = pd.read_csv('factors_dropout.csv')

# Step 1: Clean and Prepare the Data
# Map Target to binary values
data1['Target'] = data1['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Define Features and Target
X = data1.drop(columns=['Target'])
y = data1['Target']

# Separate numerical and categorical features
num_features = ['Application order', 'Age at enrollment', 'Curricular units 1st sem (credited)',
                'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)', 'Unemployment rate', 'Inflation rate', 'GDP']

cat_features = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
                'Previous qualification', 'Nacionality', 'Mother\'s qualification',
                'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation',
                'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
                'Gender', 'Scholarship holder', 'International']

# Step 2: Preprocessing Pipeline
# One-hot encode categorical variables and scale numerical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Step 3: Build Random Forest Pipeline
rf = RandomForestClassifier(random_state=42)

# Combine preprocessing and model into a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Step 4: Train the Baseline Random Forest Model
pipeline.fit(X, y)

# Cross-Validation Accuracy
cv_accuracy = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()
print("Baseline Random Forest Accuracy:", cv_accuracy)

# Step 5: Analyze Feature Importance
# Extract Feature Importances
rf_model = pipeline.named_steps['classifier']
preprocessor_transform = pipeline.named_steps['preprocessor']

# Get transformed feature names
encoded_cat_features = preprocessor_transform.named_transformers_['cat'].get_feature_names_out(cat_features)
all_features = num_features + list(encoded_cat_features)

# Plot Feature Importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display Top 15 Features
print("\nTop 15 Important Features:")
print(feature_importance_df.head(15))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.show()

# Step 6: Hyperparameter Tuning for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Random Forest Accuracy:", grid_search.best_score_)
