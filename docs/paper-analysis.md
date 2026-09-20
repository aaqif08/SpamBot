# Base Paper Analysis

**Paper:** *Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning*
**Authors:** Danish Javed, Noor Zaman Jhanjhi, Navid Ali Khan, Sayan Kumar Ray, Arafat Al-Dhaqm, Victor R. Kebande
**Venue:** IEEE Access, Vol. 13, 2025 (received 24 Jan 2025, published 18 Mar 2025)
**DOI:** 10.1109/ACCESS.2025.3551993
**License:** CC BY 4.0

This document is a structured reading of the paper written for implementation purposes. Everything under "Paper says" is a paraphrase of the paper; everything under "Implementation note" is our own engineering interpretation and is **not** a claim made by the paper.

---

## 1. Research problem

- X (Twitter) is heavily infiltrated by automated accounts (spambots, fake followers) used for spreading fake news, astroturfing, and manipulating opinion during elections.
- Most existing social-network bot detection (SNBD) systems are **black-box** models: high accuracy, but no transparency about *why* an account is flagged. This limits trust, debugging, bias detection and adaptability to evolving bots.
- Existing methods also tend to use either very large feature sets (Botometer ≈ 1,200 features) which hurt scalability, or narrow network/profile feature sets that ignore linguistic, temporal and sentiment signals.

## 2. Objectives / contributions (as stated by the paper)

1. An **interpretable** bot-detection model for spambots and fake followers on Twitter/X using SHAP and LIME.
2. Analysis of many X features and their influence on the model, evaluated across multiple bot types via well-established datasets.
3. Validation that XAI improves both **performance** and **transparency** relative to state-of-the-art baselines.
4. A **compact feature set (31 features)** that keeps performance high while remaining computationally efficient (SHAP cost grows with feature count).

## 3. Datasets

### Cresci-15 ("Fame for sale", Cresci et al. 2015) — Table 2

| Sub-dataset | Type | Accounts | Tweets |
|---|---|---|---|
| TFP (the fake project) | 100 % humans | 469 | 563,693 |
| E13 (elections 2013) | 100 % humans | 1,481 | 2,068,037 |
| FSF (fastfollowerz) | 100 % fake followers | 1,169 | 22,910 |
| INT (intertwitter) | 100 % fake followers | 1,337 | 58,925 |
| TWT (twittertechnology) | 100 % fake followers | 845 | 114,192 |

Totals: 1,950 humans / 3,351 fake followers → 5,301 accounts.

### Cresci-17 ("Paradigm shift of social spambots", Cresci et al. 2017) — Table 3

| Bot type | Description | Accounts | Tweets |
|---|---|---|---|
| Traditional spambots | classic spammers | 1,000 | 145,094 |
| Social spambots 1 | retweet an Italian political candidate | 991 | 1,610,176 |
| Social spambots 2 | spam paid mobile apps | 3,457 | 428,542 |
| Social spambots 3 | spam Amazon products | 464 | 1,418,626 |
| Fake followers | fake profiles that follow the user | 3,351 | 196,027 |
| Genuine accounts | real humans | 3,474 | 8,377,522 |

Each dataset ships as **two files per subset: `users.csv` and `tweets.csv`** (paper §III-C: "The dataset contains two distinct files which consist of user and tweet data").

**Implementation note:** the Cresci datasets are distributed by the authors (Cresci et al.) on request / via the MIB (Mining Information Bots) project. They are not bundled with this repository. The app supports importing them once obtained, and ships a clearly-labelled synthetic demo dataset otherwise.

## 4. Proposed pipeline (Figure 4)

```
Cresci dataset
  → Data preprocessing
  → Feature engineering (+ SHAP-based feature selection)
      ├─ Tweet features ──→ Sentiment analysis → Sentiment features
      └─ User features
  → Feature vector
  → Data split (Train / Val / Test, stratified)
  → Classifier (+ fine tuning via cross-validation)
  → Classification (Human / Bot)
  → Explanation (SHAP, LIME)
```

## 5. Preprocessing (§III-B)

Paper says:

- Libraries used: torch, torchtext, tqdm, **emoji**, **nltk**.
- Null handling matters for tree models. Null `description` → imputed with literal string `"missing"`; `description_length` for null descriptions kept at **0**.
- Raw text noise (symbols, URLs, mentions, emojis) is cleaned. **Emojis are converted to their textual representation** so they contribute to sentiment.
- Special characters, punctuation and whitespace are standardised/removed **only for the sentiment-analysis text path**, because URL and punctuation information are themselves used as model features.
- **Stop words removed** to shrink vocabulary and prioritise informative words.
- Word-cloud inspection (Figs 5–6) motivates cleaning: raw descriptions contain many "nan"/meaningless tokens.
- Dataset is **shuffled** before splitting.

**Implementation note:** we implement two text paths exactly as described — a *feature path* (counts of URLs, mentions, hashtags, punctuation on raw text) and a *sentiment path* (emoji→text, URL/mention removal, lower-casing, punctuation stripping, stop-word removal) before polarity/subjectivity. We use NLTK + TextBlob-style pattern polarity/subjectivity (TextBlob) for sentiment; the paper does not name the sentiment library, but `avg_polarity`/`avg_subjectivity` are TextBlob's exact output names, so TextBlob is the most faithful choice.

## 6. Feature set — Table 4 (the "31 features")

| Group | Features | Count |
|---|---|---|
| User profile | verified, friends count, followers count, listed count, favorites count | 5 |
| Content features | hashtag count, mentions count, retweet count, reply count, url count, status count | 6 |
| Engagement metrics | ffratio, avg hashtag, avg retweets, avg replies, avg mentions, avg URL, avg user engagement | 7 |
| Linguistic features | unique word count, unique word use, punctuation count, avg sentence length, punctuation density | 5 |
| Profile attributes | profile completeness, description binary, default profile, default profile image, geo enabled, profile background tile | 6 |
| Sentiment analysis | avg polarity, avg subjectivity | 2 |
| **Total** | | **31** |

Feature names as they appear in the paper's SHAP/LIME figures (Figs 7–10), which we adopt as canonical column names:
`verified, friends_count, followers_count, listed_count, favorites_count, hashtag_count, mentions_count, retweet_count, reply_count, url_count, statuses_count, ffratio, avg_hashtag, avg_retweets, avg_replies, avg_mentions, avg_url, avg_user_engagement, unique_word_count, unique_word_use, punctuation_count, avg_sentence_length, punctuation_density, profile_completeness, description_binary, default_profile, default_profile_image, geo_enabled, profile_background_tile, avg_polarity, avg_subjectivity`.

Observations from the figures:

- Figures 7–10 also show `avg_favorites`, which is **not** in Table 4. We treat it as an auxiliary feature (computed, available, but excluded from the canonical 31-feature vector by default; configurable).
- LIME value ranges such as `0.00 < ffratio <= 0.01`, `0.07 < statuses_count`, `avg_polarity <= 0.19` indicate that features were **scaled to [0, 1]** (min–max) before modelling/explanation.
- The paper does not give explicit formulas for derived features. Our formulas (documented in `docs/methodology.md`) are the standard ones:
  - `ffratio = followers_count / (friends_count + 1)`
  - `avg_X = X_count / max(statuses_count, 1)` for hashtag, retweets, replies, mentions, url, favorites
  - `avg_user_engagement = (retweet_count + reply_count + favorites_count + mentions_count) / max(statuses_count, 1)`
  - `unique_word_use = unique_word_count / total_word_count`
  - `punctuation_density = punctuation_count / total_characters`
  - `profile_completeness` = fraction of [description, location, url, profile image (non-default), profile banner/background, name, screen_name] present.
- SHAP is used both for **feature selection** (rank by mean |SHAP|, keep the compact set) and for **explanation**.

## 7. Feature selection (§III-C)

- "Shapley feature selection": train a model, compute SHAP values for every feature across the dataset, rank by mean absolute Shapley value, keep the most relevant subset. The final 31 features are the outcome of this process.
- Rationale: fewer features → lower SHAP/LIME cost, faster training/inference, competitive accuracy.

## 8. Machine-learning setup (§III, §IV)

- Classifiers compared (Tables 5–6): **Random Forest, SVM, Decision Tree, XGBoost, LightGBM, Logistic Regression, Extra Trees, Naïve Bayes, AdaBoost** (9 models).
- Hyperparameters "optimised through cross-validation" — exact grids are not published.
- Splits: the paper mentions **75 %/25 %** train/test (§IV) and also a **70 %/30 %** hold-out plus **5-fold cross-validation** (§IV-A). Tables 5–6 are stated to come from **5-fold CV**.
- **Stratified** splitting to preserve class ratio.
- Binary target: Human (0) vs Bot (1). All bot sub-types are merged into the positive class.

**Implementation note:** we default to stratified 75/25 hold-out + stratified 5-fold CV on the training portion, and a small randomised hyperparameter search per model. Both split sizes and fold count are user-configurable in the training UI.

## 9. Evaluation metrics (§IV-A)

Accuracy, Precision, Recall, F1 (Eq. 1), AUC (ROC), plus a qualitative "Interpretability" criterion (ability to explain decisions with SHAP/LIME). We additionally compute the confusion matrix, ROC curve and precision-recall curve since they underpin the metrics reported.

## 10. Explainability

### LIME (§III-D, Figs 7–8)
- Local, model-agnostic; per-account explanation.
- Output: prediction probabilities (Human vs Bot), two-column list of feature-range rules with contribution weights toward "Human" (left) or "Bot" (right).
- Example insights quoted by the paper: high `reply_count` → bot (automated engagement); hashtag usage → bot (+0.11); low follower/following ratio → bot; medium `retweet_count` → human; `avg_mentions > 0`, `favorites_count > 0` → human; `default_profile` → bot.

### SHAP (§III-E, Figs 9–10)
- Global explanation via **beeswarm summary plots** of the **top 20** features; y-axis ordered by mean |SHAP|; x-axis = SHAP value (impact on model output); colour = feature value.
- Higher SHAP value → higher likelihood of malicious behaviour.
- Top features: Cresci-15 — `favorites_count, avg_sentence_length, avg_favorites, verified, statuses_count, listed_count, geo_enabled, followers_count, avg_mentions, punctuation_count …`; Cresci-17 — `ffratio, friends_count, mentions_count, avg_replies, hashtag_count, avg_mentions, url_count, avg_favorites, reply_count, followers_count …`.
- Discussion highlights `favorites_count, avg_mentions, unique_word_use, ffratio` as high-value features.

## 11. Reported results (paper's own experiments — 5-fold CV)

> **These numbers are reported by the base paper. They are NOT results of this implementation.** The application displays them only under the label "Reported in base paper".

### Table 5 — Cresci-15

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.990 | 0.994 | 0.990 | 0.992 | 0.999 |
| SVM | 0.959 | 0.972 | 0.962 | 0.967 | 0.983 |
| Decision Tree | 0.976 | 0.980 | 0.982 | 0.981 | 0.974 |
| XGBoost | 0.991 | 0.994 | 0.991 | 0.993 | 0.999 |
| LightGBM | 0.991 | 0.994 | 0.992 | 0.993 | 0.999 |
| Logistic Regression | 0.954 | 0.973 | 0.953 | 0.963 | 0.977 |
| Extra Trees | 0.987 | 0.994 | 0.986 | 0.990 | 0.999 |
| Naïve Bayes | 0.768 | 0.739 | 0.980 | 0.842 | 0.966 |
| AdaBoost | 0.986 | 0.991 | 0.988 | 0.990 | 0.998 |

### Table 6 — Cresci-17

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.988 | 0.992 | 0.992 | 0.992 | 0.998 |
| SVM | 0.964 | 0.981 | 0.972 | 0.977 | 0.991 |
| Decision Tree | 0.984 | 0.989 | 0.989 | 0.989 | 0.978 |
| XGBoost | 0.990 | 0.994 | 0.993 | 0.993 | 0.999 |
| LightGBM | 0.990 | 0.994 | 0.993 | 0.993 | 0.999 |
| Logistic Regression | 0.939 | 0.964 | 0.955 | 0.981 | 0.981 |
| Extra Trees | 0.987 | 0.991 | 0.992 | 0.991 | 0.998 |
| Naïve Bayes | 0.919 | 0.992 | 0.899 | 0.944 | 0.979 |
| AdaBoost | 0.983 | 0.989 | 0.990 | 0.989 | 0.997 |

### Baseline comparison (Tables 7–8, as compiled by the paper)
- Cresci-15: best baseline 0.988 acc / 0.988 F1 ([52]); paper's LightGBM 0.991 / 0.993.
- Cresci-17: best baseline 0.985 acc / 0.989 F1 ([34]); paper's XGBoost 0.990 / 0.993.

## 12. Limitations acknowledged by the paper

- SHAP has exponential worst-case complexity; LIME trains a surrogate per instance — both are expensive at Cresci scale. Mitigated by the compact feature set.
- Reliance on a fixed feature set may not transfer to new-generation bots that mimic humans.
- Non-interpretable baselines cannot adapt/debug; but even interpretable models depend on static features that may lose relevance.

## 13. Future scope (paper)

- Adaptive / continual learning to follow evolving bot behaviour.
- Graph neural networks over the social graph combined with XAI.

## 14. What this repository will and will not claim

| Item | Status |
|---|---|
| 31-feature set, six groups | Implemented exactly (Table 4) |
| Two-path text preprocessing (feature path vs sentiment path) | Implemented |
| Emoji → text, stop-word removal, "missing" description imputation | Implemented |
| 9 classifiers | Implemented |
| Stratified split, 5-fold CV, hyperparameter search | Implemented (configurable) |
| SHAP global (beeswarm data, mean |SHAP|) + local (waterfall data) | Implemented with real SHAP |
| LIME local explanation with Human/Bot columns | Implemented with real LIME |
| SHAP-based feature selection | Implemented as an optional training step (rank by mean |SHAP|, keep top-k) |
| Cresci-15 / Cresci-17 numbers | Displayed only as "Reported in base paper"; our own numbers displayed separately as "Reproduced by this implementation" and only when a real training run exists |
| Real X/Twitter API | Not assumed; adapter interface + clearly labelled sample adapter only |
