<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=FF4B4B&center=true&vCenter=true&width=600&lines=%E2%9D%A4%EF%B8%8F+Heart+Disease+Prediction;ML+Pipeline+%E2%80%94+5+Models;End-to-End+Healthcare+AI" alt="Heart Disease Prediction" />

<br>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Dataset-UCI%20Heart-005A9C?style=for-the-badge&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&labelColor=1a1a2e">

<br>

<img src="https://github-readme-stats.vercel.app/api?username=basitali08&show_icons=true&theme=radical&hide_border=true&count_private=true" width="400">

</div>

---

## Overview

Complete ML pipeline comparing **5 models** for heart disease prediction — from preprocessing to evaluation to model serialization.

---

## Methodology

| Step | Detail |
|------|--------|
| **Preprocessing** | Missing value imputation, StandardScaler, train/test split |
| **Models** | Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC AUC, Cross-validation |
| **Output** | Trained model + preprocessor serialized for deployment |

---

## Results

| Model | Accuracy |
|-------|:--------:|
| Gradient Boosting | 52.46% |
| KNN | 44.26% |
| Random Forest | 44.26% |
| SVM | 40.98% |
| Logistic Regression | 36.07% |

> **Note**: This project was built with synthetic/small data as an ML pipeline demo. For the tuned version with **91.8% accuracy**, see [Advanced ML with Optuna + SHAP](https://github.com/basitali08/-advanced-ml-heart-disease).

---

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

---

## Quick Start

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python src/heart_disease_pipeline.py
```

---

## Use Saved Model

```python
import pickle
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))
model = pickle.load(open('models/best_model.pkl', 'rb'))
prediction = model.predict(preprocessor.transform(new_data))
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| Scikit-learn | ML models & preprocessing |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |

---

<div align="center">

**Built with Python, Scikit-learn, Pandas**

[![GitHub stars](https://img.shields.io/github/stars/basitali08/heart-disease-prediction?style=social)](https://github.com/basitali08/heart-disease-prediction)
[![GitHub forks](https://img.shields.io/github/forks/basitali08/heart-disease-prediction?style=social)](https://github.com/basitali08/heart-disease-prediction)

</div>

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a><br>
<sub>Healthcare Machine Learning · MS Data Science Portfolio</sub>
</p>