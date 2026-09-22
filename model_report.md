# EasyVisa - Model Report

**Objective:** classify US employment-based visa applications as *Certified* or *Denied*, and identify the factors that drive certification, so the Office of Foreign Labor Certification (OFLC) can prioritize review and facilitate qualified approvals.

**Primary metric:** Recall. The costliest error is a **false negative**: a genuinely qualified applicant predicted as denied, which means a deserving candidate loses an opportunity and OFLC fails its mandate. Recall was therefore the objective for cross-validation, tuning and final selection, with F1, precision and accuracy used as tie-breakers.

All numbers below are produced by `scripts/train.py` on the real dataset (`random_state=1`) and stored in `reports/metrics.json`.

## Data

- 25,480 applications, 11 predictors after dropping the identifier `case_id`.
- Target distribution: 66.8% Certified, 33.2% Denied.
- Cleaning: 33 negative `no_of_employees` values corrected to their absolute value (data-entry sign errors).
- Encoding: target mapped to 1/0; categorical predictors one-hot encoded (`drop_first=True`) into 21 features.
- Split (stratified): train 17,836 · validation 6,879 · test 765.

## Model comparison (5-fold cross-validated recall)

Six classifiers were spot-checked on three data variants: original, SMOTE-oversampled, and random-undersampled.

| Model | Original | Oversampled | Undersampled |
| --- | --- | --- | --- |
| AdaBoost | **0.887** | **0.862** | 0.713 |
| GBM | 0.873 | 0.859 | **0.719** |
| XGBoost | 0.853 | 0.844 | 0.697 |
| Random Forest | 0.839 | 0.819 | 0.683 |
| Bagging | 0.775 | 0.744 | 0.604 |
| Decision Tree | 0.740 | 0.719 | 0.618 |

AdaBoost and Gradient Boosting were the strongest and most stable, so both were carried forward for tuning.

## Hyperparameter tuning

`RandomizedSearchCV`, recall-scored, 5-fold, `n_iter=50`.

**AdaBoost (oversampled)**: best CV recall **0.932**
`n_estimators=75, learning_rate=0.01, estimator=DecisionTree(max_depth=1)`

**Gradient Boosting (undersampled)**: best CV recall **0.903**
`n_estimators=100, learning_rate=0.005, subsample=0.9, max_features='log2', init=DecisionTree(max_depth=1)`

## Tuned model performance

| Model | Set | Accuracy | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- |
| AdaBoost (oversampled) | Validation | 0.706 | 0.930 | 0.715 | 0.808 |
| Gradient Boosting (undersampled) | Validation | 0.726 | 0.898 | 0.745 | 0.814 |
| **Gradient Boosting (undersampled)** | **Test** | **0.739** | **0.900** | **0.755** | **0.821** |

## Final model

**Gradient Boosting tuned on undersampled data.** It edges out AdaBoost on F1, precision and accuracy while holding recall at ~0.90, and its test performance (recall 0.900) is in line with its train and validation scores, indicating a generalized, non-overfit model. The trade-off is deliberate: with recall as the business priority, the model accepts lower precision to avoid missing qualified applicants.

The trained model is persisted at `models/gradient_boosting_easyvisa.joblib`.

## Key drivers of certification (feature importance)

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | Prior job experience | 0.27 |
| 2 | Prevailing wage | 0.17 |
| 3 | Yearly wage unit | 0.16 |
| 4 | Master's degree | 0.11 |
| 5 | Continent: Europe | 0.10 |
| 6 | Doctorate degree | 0.06 |
| 7 | Region: Midwest | 0.04 |
| 8 | Region: West | 0.03 |

## Recommendations

- **OFLC:** deploy the model to auto-shortlist high-probability cases and focus manual review where the model is uncertain, cutting turnaround without lowering standards.
- **Employers:** prioritize experienced, advanced-degree candidates and structure compensation as annual salaries, hourly-wage applications are certified far less often.
- **Applicants:** foreground prior experience and the highest qualification held.

## Reproducing these results

```bash
pip install -r requirements.txt
python scripts/train.py            # writes metrics.json, figures, and the model
```
