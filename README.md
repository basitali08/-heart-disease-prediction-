# Heart Disease Prediction Model

## Overview
A complete machine learning pipeline to predict heart disease presence using clinical attributes. This project demonstrates end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment readiness.

## Dataset
UCI Heart Disease Dataset (Cleveland) - 303 patient records with 13 medical attributes

## Methodology
- **Preprocessing**: Missing value imputation, feature scaling
- **Models**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-validation

## Results
| Model | Accuracy |
|-------|----------|
| Gradient Boosting | 52.46% |
| KNN | 44.26% |
| Random Forest | 44.26% |
| SVM | 40.98% |
| Logistic Regression | 36.07% |

## Key Features Correlated with Heart Disease
1. Max Heart Rate (thalach)
2. Resting Blood Pressure (trestbps)
3. Age
4. ST Depression (oldpeak)
5. Cholesterol (chol)

## Project Structure
```
heart-disease-prediction/
├── data/raw/              # Raw dataset
├── data/processed/        # Processed data
├── src/                   # Source code
│   └── heart_disease_pipeline.py  # Complete ML pipeline
├── models/                # Saved model artifacts
├── results/               # Evaluation results and plots
└── README.md
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python src/heart_disease_pipeline.py
```

## Use Saved Model
```python
import pickle
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))
model = pickle.load(open('models/best_model.pkl', 'rb'))
prediction = model.predict(preprocessor.transform(new_data))
```

## Next Steps
- Hyperparameter tuning for improved accuracy
- Model deployment via Flask API
- Integration with real clinical data (MIMIC-III)
