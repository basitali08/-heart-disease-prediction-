# ❤️ Heart Disease Prediction — ML Pipeline (5 Models)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)](https://pandas.pydata.org)
[![UCI](https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-005A9C)](https://archive.ics.uci.edu/ml/datasets/heart+disease)

Complete ML pipeline comparing 5 models for heart disease prediction — from preprocessing to evaluation to model serialization.

---

## Methodology

| Step | Detail |
|------|--------|
| **Preprocessing** | Missing value imputation, StandardScaler, train/test split |
| **Models** | Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC AUC, Cross-validation |
| **Output** | Trained model + preprocessor serialized for deployment |

## Results

| Model | Accuracy |
|-------|:--------:|
| Gradient Boosting | 52.46% |
| KNN | 44.26% |
| Random Forest | 44.26% |
| SVM | 40.98% |
| Logistic Regression | 36.07% |

> **Note**: This project was built with synthetic/small data as an ML pipeline demo. For the tuned version with **91.8% accuracy**, see [Advanced ML with Optuna + SHAP](https://github.com/basitali08/-advanced-ml-heart-disease).

## Project Structure

```
heart-disease-prediction/
├── src/
│   └── heart_disease_pipeline.py   # Complete pipeline (5 models)
├── data/                            # Dataset (raw + processed)
├── models/                          # Serialized model + preprocessor
├── results/                         # Evaluation metrics + plots
├── requirements.txt
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

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a>
</p>
