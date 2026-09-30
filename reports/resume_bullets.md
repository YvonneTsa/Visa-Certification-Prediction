# Resume & Portfolio Bullet Points

Copy-ready lines describing this project, tailored for data-science and fintech roles. Pick the version that fits the space you have.

## Why this project is relevant to fintech

Visa certification is an imbalanced, high-stakes binary approval decision with an explicit cost asymmetry between the two error types. That is the same shape as the core decisioning problems in fintech: loan and credit approval, KYC and onboarding risk, fraud and AML alert triage, and transaction authorization. The techniques used here transfer directly: class-imbalance handling (SMOTE and undersampling), recall-vs-precision trade-off management, threshold and metric selection driven by business cost, ensemble modeling, and model interpretability for decisions that must be explainable to reviewers and regulators.

## Resume (concise)

- Built an ensemble machine-learning pipeline to predict work-visa certification on 25,480 imbalanced applications, reaching **90% recall** on a held-out test set with a tuned Gradient Boosting model.
- Compared six classifiers (Random Forest, Gradient Boosting, AdaBoost, XGBoost, Bagging, Decision Tree) across original, SMOTE-oversampled and undersampled data using 5-fold stratified cross-validation.
- Optimized for **recall** to minimize costly false negatives, tuned top models with RandomizedSearchCV, and used feature importance to explain the decision drivers (prior experience, prevailing wage, education).
- Packaged the work as a reproducible, production-style repository: modular Python package, a one-command training pipeline, a persisted model artifact, and generated reports.

## Resume (one-liner)

- Developed a recall-optimized Gradient Boosting classifier for an imbalanced approval-decisioning problem (90% test recall), shipped as a reproducible, documented pipeline.

## Portfolio / LinkedIn (expanded)

**EasyVisa: Approval-Decisioning with Ensemble Machine Learning**
Built an end-to-end supervised-learning system to predict US work-visa certification, framed as an imbalanced approval-decisioning problem directly analogous to fintech credit, KYC and fraud decisioning. Performed EDA on 25,480 applications, handled class imbalance with SMOTE and random undersampling, and benchmarked six ensemble classifiers with stratified cross-validation. Optimized for recall because the costliest error is rejecting a qualified applicant, then tuned AdaBoost and Gradient Boosting with RandomizedSearchCV. The final Gradient Boosting model generalized to 90% recall on the test set. Feature-importance analysis identified prior job experience, prevailing wage, yearly wage structure and advanced education as the dominant drivers, translated into recommendations for the reviewing body, employers and applicants. Shipped as a reproducible repository with a modular Python package, a one-command training pipeline, a persisted model, and an executive report.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Matplotlib, Seaborn, joblib

## Design + data angle (for a hybrid UX/data role)

For a role that blends product design and data science, this project shows the full loop: not only building the model that scores an approval decision, but understanding the cost trade-offs and interpretability that a reviewer-facing decisioning tool would need. It pairs naturally with designing the alert-triage or case-review interface that sits on top of a model like this.

## Skills demonstrated

Data cleaning · Exploratory data analysis · Feature encoding · Class-imbalance handling (SMOTE / undersampling) · Ensemble methods · Cross-validation · Hyperparameter tuning · Cost-sensitive metric selection · Model evaluation & selection · Model interpretability · Reproducible ML engineering · Business insight communication
