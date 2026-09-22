# EasyVisa - US Work Visa Approval Prediction

Predicting whether a US employment-based visa (labor certification) application will be **Certified** or **Denied**, and identifying the profile drivers behind approval, using ensemble machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12.3-teal)
![License](https://img.shields.io/badge/License-MIT-green)

> **Final model, Gradient Boosting (tuned, undersampled): 90.0% recall on the held-out test set.**

---

## Overview

The Office of Foreign Labor Certification (OFLC) processes hundreds of thousands of employer applications each year to bring foreign workers into the United States. In FY 2016 alone it handled **775,979 applications** covering roughly **1.7 million positions**, a nine percent year-over-year increase. Reviewing every case manually is slow and does not scale.

This project builds a supervised classification model that helps OFLC **shortlist applications with a high likelihood of certification** and surfaces the factors that most influence the decision, so review can be faster and more consistent without lowering labor standards. The work is framed from the perspective of a data scientist at **EasyVisa**, the firm engaged by OFLC.

Every figure and metric in this repository is generated from the real data by a reproducible pipeline (`scripts/train.py`) and stored in [`reports/metrics.json`](reports/metrics.json).

### Why this problem matters beyond visas

Visa certification is an imbalanced, high-stakes binary approval decision with an explicit cost asymmetry between the two error types. That is the same shape as the core decisioning problems in **fintech**: loan and credit approval, KYC and onboarding risk, and fraud or AML alert triage. The methods applied here transfer directly to those settings: class-imbalance handling, a recall-versus-precision trade-off driven by business cost, ensemble modeling, and interpretable drivers for decisions that reviewers and regulators must be able to explain.

## Problem framing

- **Task:** binary classification, `Certified` (1) vs `Denied` (0).
- **Primary metric: recall.** The costliest error is a **false negative**: flagging a genuinely qualified applicant as "deny." That means a deserving candidate loses an opportunity and OFLC fails its mandate to facilitate valid approvals. Maximizing recall minimizes those missed certifications; F1, precision and accuracy are used as tie-breakers.

## Results

Six classifiers were compared with 5-fold stratified cross-validation across three data variants (original, SMOTE-oversampled, random-undersampled). AdaBoost and Gradient Boosting were the strongest and were tuned with `RandomizedSearchCV`.

| Model | Set | Accuracy | Recall | Precision | F1 |
| --- | --- | :---: | :---: | :---: | :---: |
| AdaBoost (oversampled, tuned) | Validation | 0.706 | 0.930 | 0.715 | 0.808 |
| Gradient Boosting (undersampled, tuned) | Validation | 0.726 | 0.898 | 0.745 | 0.814 |
| **Gradient Boosting (undersampled, tuned)** | **Test** | **0.739** | **0.900** | **0.755** | **0.821** |

The final Gradient Boosting model generalizes cleanly, its test recall (0.900) matches its train and validation scores. Full breakdown in [`reports/model_report.md`](reports/model_report.md).

<p align="center">
  <img src="reports/figures/model_comparison_recall.png" width="48%" alt="Cross-validated recall by model">
  <img src="reports/figures/confusion_matrix_test.png" width="42%" alt="Confusion matrix on the test set">
</p>

### Key drivers of certification

![Feature importance](reports/figures/feature_importance.png)

1. **Prior job experience**: the single strongest predictor.
2. **Prevailing wage**: higher wages are strongly associated with certification.
3. **Yearly wage unit**: annual salary structures are certified far more often than hourly.
4. **Advanced education**: Master's and Doctorate degrees substantially raise approval odds.
5. **Continent & region**: applicants from Europe and jobs in the Midwest/West see a meaningful boost, though profile factors dominate.

### Recommendations

- **OFLC:** auto-shortlist and prioritize high-probability cases; focus manual review where the model is uncertain.
- **Employers:** invest in experienced, advanced-degree applicants and structure compensation as annual salaries.
- **Applicants:** foreground prior experience and the highest qualification held.

## Dataset

25,480 visa applications, 12 columns (employee and employer attributes). Target: `case_status`. The `EasyVisa.csv` file is included in [`data/`](data/) so the project runs out of the box; the full data dictionary is in [`data/README.md`](data/README.md).

## Approach

1. **Data cleaning**: corrected negative `no_of_employees` values (data-entry sign errors), dropped the non-predictive `case_id`.
2. **Exploratory data analysis**: univariate and bivariate analysis of every feature against `case_status` with reusable plotting helpers.
3. **Preprocessing**: target encoded to 0/1, one-hot encoding of categoricals, stratified 70/30 train-validation split then a further 90/10 validation-test split.
4. **Model spot-check**: six classifiers × three sampling strategies, 5-fold stratified CV, recall-scored.
5. **Hyperparameter tuning**: `RandomizedSearchCV` (recall-scored, 5-fold, `n_iter=50`) on AdaBoost and Gradient Boosting.
6. **Final selection & evaluation**: best model confirmed on the held-out test set; feature importances extracted for interpretability; model persisted to `models/`.

## Repository structure

```
easyvisa-approval-prediction/
├── README.md
├── requirements.txt
├── pyproject.toml               # installable package (src/easyvisa)
├── config.yaml                  # paths & pipeline parameters
├── Makefile                     # make install / train / notebook
├── LICENSE
├── .gitignore
├── data/
│   ├── EasyVisa.csv             # dataset (25,480 rows)
│   └── README.md                # data dictionary
├── notebooks/
│   └── easyvisa_approval_prediction.ipynb   # full executed analysis (EDA → modeling → insights)
├── src/easyvisa/
│   ├── data.py                  # load & clean
│   ├── preprocess.py            # encode & stratified splits
│   ├── plots.py                 # reusable EDA visualizations
│   ├── models.py                # model zoo, sampling, CV, tuning
│   └── evaluate.py              # metrics, confusion matrix, feature importance
├── scripts/
│   └── train.py                 # reproducible end-to-end pipeline
├── models/
│   └── gradient_boosting_easyvisa.joblib     # persisted final model
└── reports/
    ├── easyvisa_approval_prediction.html     # rendered notebook (all outputs & charts)
    ├── model_report.md          # executive model report
    ├── metrics.json             # verified metrics for every stage
    ├── resume_bullets.md        # ready-to-use portfolio bullet points
    └── figures/                 # generated charts
```

## Getting started

```bash
# 1. Clone
git clone https://github.com/<your-username>/easyvisa-approval-prediction.git
cd easyvisa-approval-prediction

# 2. (Recommended) virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt        # or: make install

# 4a. Reproduce the full pipeline (metrics, figures, saved model)
python scripts/train.py                 # or: make train

# 4b. Or explore the analysis notebook
jupyter notebook notebooks/easyvisa_approval_prediction.ipynb   # or: make notebook
```

Using the trained model:

```python
import joblib
model = joblib.load("models/gradient_boosting_easyvisa.joblib")
model.predict(X_new)   # X_new: one-hot encoded like the training features
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE, RandomUnderSampler) · Matplotlib · Seaborn · joblib

## License

Released under the [MIT License](LICENSE).

---

*Completed as part of graduate coursework in supervised learning and ensemble techniques, then refactored into a reproducible, production-style project. The approach carries over to imbalanced approval and risk-decisioning problems common in fintech.*
