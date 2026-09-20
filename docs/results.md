# Experimental results — reproduced by this implementation

Generated 2026-09-19 04:41 UTC by `scripts/report_results.py` (development run on the public Cresci mirrors, before the multi-tenant platform conversion; the script now reads models from the database of the organization given with `--email`). These numbers are a record of what the method achieved on that data with the listed configuration — they are not shown in the application unless the same models are trained there.

All numbers in sections 1–2 were **measured by this implementation** on the real Cresci datasets (user-level mirror, see `backend/data/datasets/PROVENANCE.md`). They are **not** the paper's numbers; the paper's numbers appear only in section 3.

Protocol: shuffle → stratified 75/25 hold-out → randomised hyperparameter search (F1, stratified 5-fold CV) on the training split → 5-fold CV of the tuned pipeline → hold-out evaluation. Tweet-derived features are constant without `tweets.csv` and are dropped automatically; the number of features actually used is shown per model.

## 1. Hold-out and cross-validation results per dataset

### CRESCI-17 (imported from local files)

| Classifier | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean ± std) | CV AUC (mean ± std) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 20 | 0.986 | 0.993 | 0.987 | 0.990 | 0.997 | 0.990 ± 0.002 | 0.997 ± 0.001 | 22.1s |
| SVM | 20 | 0.934 | 0.947 | 0.964 | 0.955 | 0.974 | 0.958 ± 0.003 | 0.973 ± 0.005 | 86.2s |
| Decision Tree | 20 | 0.979 | 0.987 | 0.984 | 0.986 | 0.992 | 0.988 ± 0.002 | 0.992 ± 0.003 | 0.7s |
| XGBoost | 20 | 0.986 | 0.993 | 0.987 | 0.990 | 0.997 | 0.991 ± 0.002 | 0.997 ± 0.001 | 7.4s |
| LightGBM | 20 | 0.986 | 0.994 | 0.987 | 0.991 | 0.997 | 0.991 ± 0.002 | 0.997 ± 0.001 | 5.7s |
| Logistic Regression | 20 | 0.935 | 0.935 | 0.978 | 0.956 | 0.966 | 0.955 ± 0.001 | 0.968 ± 0.002 | 0.7s |
| Extra Trees | 20 | 0.962 | 0.965 | 0.983 | 0.974 | 0.993 | 0.975 ± 0.003 | 0.993 ± 0.001 | 8.8s |
| Naive Bayes | 20 | 0.906 | 0.902 | 0.977 | 0.938 | 0.933 | 0.940 ± 0.001 | 0.942 ± 0.004 | 0.3s |
| AdaBoost | 20 | 0.983 | 0.988 | 0.989 | 0.988 | 0.996 | 0.989 ± 0.003 | 0.997 ± 0.001 | 22.3s |

### CRESCI-15 (imported from local files)

| Classifier | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean ± std) | CV AUC (mean ± std) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 19 | 0.977 | 0.990 | 0.973 | 0.981 | 0.997 | 0.987 ± 0.002 | 0.998 ± 0.001 | 10.2s |
| SVM | 19 | 0.900 | 0.903 | 0.943 | 0.922 | 0.970 | 0.927 ± 0.006 | 0.959 ± 0.017 | 9.8s |
| Decision Tree | 19 | 0.969 | 0.980 | 0.971 | 0.975 | 0.973 | 0.983 ± 0.004 | 0.981 ± 0.006 | 0.3s |
| XGBoost | 19 | 0.982 | 0.990 | 0.981 | 0.986 | 0.997 | 0.988 ± 0.003 | 0.998 ± 0.001 | 3.8s |
| LightGBM | 19 | 0.981 | 0.990 | 0.980 | 0.985 | 0.998 | 0.987 ± 0.002 | 0.998 ± 0.001 | 3.0s |
| Logistic Regression | 19 | 0.882 | 0.887 | 0.932 | 0.909 | 0.933 | 0.920 ± 0.006 | 0.932 ± 0.007 | 0.3s |
| Extra Trees | 19 | 0.956 | 0.980 | 0.950 | 0.965 | 0.991 | 0.972 ± 0.008 | 0.994 ± 0.001 | 7.0s |
| Naive Bayes | 19 | 0.882 | 0.887 | 0.931 | 0.909 | 0.886 | 0.919 ± 0.005 | 0.900 ± 0.005 | 0.2s |
| AdaBoost | 19 | 0.978 | 0.987 | 0.979 | 0.983 | 0.995 | 0.985 ± 0.004 | 0.998 ± 0.001 | 11.7s |

### CRESCI-15 + CRESCI-17 combined (user level)

| Classifier | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean ± std) | CV AUC (mean ± std) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM **(active)** | 20 | 0.980 | 0.987 | 0.981 | 0.984 | 0.995 | 0.986 ± 0.002 | 0.996 ± 0.002 | 12.0s |

## 2. Cross-dataset generalisation (out-of-distribution)

For each Cresci dataset, the classifier with the highest hold-out F1 trained on that dataset is evaluated on the **other** dataset, which it never saw. The fake-follower accounts overlap between the two collections, so this is a partial rather than fully disjoint test; the row for the active model is in-sample when its training data includes the evaluation set.

| Model | Evaluated on | Scope | Accounts | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM trained on cresci-17 | cresci-15 | out-of-distribution | 5,301 | 0.929 | 0.901 | 0.998 | 0.947 | 0.994 | TN 1584 · FP 366 · FN 8 · TP 3343 |
| XGBoost trained on cresci-15 | cresci-17 | out-of-distribution | 12,737 | 0.588 | 0.993 | 0.437 | 0.607 | 0.949 | TN 3445 · FP 29 · FN 5218 · TP 4045 |
| LightGBM (active, trained on CRESCI-15 + CRESCI-17 combined (user level)) | cresci-15 | in-sample | 5,301 | 0.991 | 0.990 | 0.995 | 0.993 | 0.999 | TN 1916 · FP 34 · FN 16 · TP 3335 |
| LightGBM (active, trained on CRESCI-15 + CRESCI-17 combined (user level)) | cresci-17 | in-sample | 12,737 | 0.995 | 1.000 | 0.993 | 0.996 | 0.999 | TN 3471 · FP 3 · FN 66 · TP 9197 |

## 3. Reported in base paper (reference only — NOT measured here)

Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning, IEEE Access, vol. 13, pp. 52246–52259 (2025), DOI 10.1109/ACCESS.2025.3551993. 5-fold CV with the full 31 features including tweet-derived ones.

### cresci-15 (Table 5)

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.990 | 0.994 | 0.990 | 0.992 | 0.999 |
| SVM | 0.959 | 0.972 | 0.962 | 0.967 | 0.983 |
| Decision Tree | 0.976 | 0.980 | 0.982 | 0.981 | 0.974 |
| XGBoost | 0.991 | 0.994 | 0.991 | 0.993 | 0.999 |
| LightGBM | 0.991 | 0.994 | 0.992 | 0.993 | 0.999 |
| Logistic Regression | 0.954 | 0.973 | 0.953 | 0.963 | 0.977 |
| Extra Trees | 0.987 | 0.994 | 0.986 | 0.990 | 0.999 |
| Naive Bayes | 0.768 | 0.739 | 0.980 | 0.842 | 0.966 |
| AdaBoost | 0.986 | 0.991 | 0.988 | 0.990 | 0.998 |

### cresci-17 (Table 6)

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.988 | 0.992 | 0.992 | 0.992 | 0.998 |
| SVM | 0.964 | 0.981 | 0.972 | 0.977 | 0.991 |
| Decision Tree | 0.984 | 0.989 | 0.989 | 0.989 | 0.978 |
| XGBoost | 0.990 | 0.994 | 0.993 | 0.993 | 0.999 |
| LightGBM | 0.990 | 0.994 | 0.993 | 0.993 | 0.999 |
| Logistic Regression | 0.939 | 0.964 | 0.955 | 0.981 | 0.981 |
| Extra Trees | 0.987 | 0.991 | 0.992 | 0.991 | 0.998 |
| Naive Bayes | 0.919 | 0.992 | 0.899 | 0.944 | 0.979 |
| AdaBoost | 0.983 | 0.989 | 0.990 | 0.989 | 0.997 |

## 4. Why our numbers differ from the paper

- The public mirror contains only account profiles; the 11 tweet-derived features (hashtag/mention/URL/retweet/reply counts and their per-tweet averages, engagement) are unavailable and were dropped. Linguistic and sentiment features are computed from the profile description instead of tweets.
- Hyperparameter search spaces and derived-feature formulas are ours (not published in the paper).
- Paper results are 5-fold CV averages; our headline numbers are a stratified 25% hold-out (CV means are reported alongside).
