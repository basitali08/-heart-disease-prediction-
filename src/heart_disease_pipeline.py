import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
for d in ['data/processed', 'models', 'results']:
    os.makedirs(d, exist_ok=True)

print('='*60)
print('HEART DISEASE PREDICTION PIPELINE')
print('='*60)

# 1. Load Data
print('\n1. LOADING DATA...')
df = pd.read_csv('data/raw/heart_disease.csv')
print(f'Dataset shape: {df.shape}')
print(f'Features: {list(df.columns)}')

# 2. Initial Exploration
print('\n2. DATA OVERVIEW...')
print(df.info())
print(f'\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}')

# 3. Preprocessing
print('\n3. PREPROCESSING...')
X = df.drop('target', axis=1)
y = df['target']

numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print(f'Training set: {X_train_processed.shape}')
print(f'Test set: {X_test_processed.shape}')

# 4. Model Training & Evaluation
print('\n4. TRAINING MODELS...')
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    cv_score = cross_val_score(model, X_train_processed, y_train, cv=5).mean()
    
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'ROC AUC': round(auc, 4),
        'CV Score': round(cv_score, 4)
    })
    
    print(f'  {name:20s} | Acc: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = name

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# 5. Results Summary
print('\n5. RESULTS SUMMARY...')
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('results/model_comparison.csv', index=False)

# 6. Feature Importance (Random Forest)
print('\n6. FEATURE IMPORTANCE...')
rf = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': numerical_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))
feature_importance.to_csv('results/feature_importance.csv', index=False)

# 7. Best Model Evaluation
print(f'\n7. BEST MODEL: {best_model_name}')
y_pred_best = best_model.predict(X_test_processed)
print(f'\nClassification Report:\n{classification_report(y_test, y_pred_best)}')

cm = confusion_matrix(y_test, y_pred_best)
print(f'Confusion Matrix:\n{cm}')

# 8. Visualizations
print('\n8. CREATING VISUALIZATIONS...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Model comparison
ax = axes[0, 0]
bars = ax.bar(results_df['Model'], results_df['Accuracy'], color='steelblue')
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, results_df['Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

# Feature importance
ax = axes[0, 1]
top_features = feature_importance.head(8)
bars = ax.barh(top_features['Feature'], top_features['Importance'], color='coral')
ax.set_title('Top 8 Feature Importances', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
ax.invert_yaxis()

# Confusion Matrix
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# ROC Curves
ax = axes[1, 1]
from sklearn.metrics import roc_curve
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/performance_plots.png', dpi=200, bbox_inches='tight')
print('Plots saved to results/performance_plots.png')

print('\n' + '='*60)
print('PIPELINE COMPLETE!')
print('='*60)
print(f'\nBest Model: {best_model_name}')
print(f'Best Accuracy: {best_score:.4f}')
print(f'Models saved to: models/')
print(f'Results saved to: results/')
print(f'To use the model:')
print('  1. Load preprocessor: pickle.load(open("models/preprocessor.pkl", "rb"))')
print('  2. Load model: pickle.load(open("models/best_model.pkl", "rb"))')
print('  3. Preprocess new data: preprocessor.transform(new_data)')
print('  4. Predict: model.predict(processed_data)')