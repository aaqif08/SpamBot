# Methodology — how the implementation maps to the base paper

Base paper: Javed et al., *Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning*, IEEE Access 13 (2025), DOI 10.1109/ACCESS.2025.3551993.

This document states exactly what was implemented, which formulas were chosen where the paper is silent, and every engineering adaptation. Nothing in the paper's methodology was changed silently.

## 1. Pipeline

| Paper stage (Fig. 4) | Implementation |
|---|---|
| Cresci dataset | `ml/datasets.py` — importer for local Cresci-15 / Cresci-17 folders (`users.csv` + `tweets.csv` per subset); labelled/unlabelled CSV inspection; synthetic demo generator (`ml/demo_data.py`) |
| Data preprocessing | `ml/preprocessing.py` — two text paths (feature path / sentiment path), `"missing"` imputation, emoji→text, stop-word removal; numeric median imputation + min–max scaling inside the sklearn pipeline |
| Feature engineering / selection | `ml/features.py` (`FEATURE_GROUPS`, `FeatureExtractor`); SHAP-based selection in `ml/train.py` |
| Tweet features / user features | `FeatureExtractor.extract_content_features`, `extract_linguistic_features`, `extract_user_features`, `extract_profile_features` |
| Sentiment analysis | `ml/sentiment.py` (TextBlob PatternAnalyzer) |
| Feature vector | 31 columns in the canonical order of `FEATURE_NAMES` |
| Train / Val / Test split | stratified hold-out (`test_size`, default 0.25) + stratified k-fold CV on the training portion (default 5) |
| Classifier + fine tuning | nine classifiers, `RandomizedSearchCV` scored by F1 |
| Classification | `ml/predict.py` — probability, label, risk score |
| Explanation | `ml/explain.py` — SHAP (global/local) and LIME (local) |

## 2. Preprocessing (paper §III-B)

Feature path (raw text is preserved because URLs and punctuation are features):

- URL count: regex over `http(s)://`, `www.` and common TLD patterns.
- Mention count: `@handle` tokens; hashtag count: `#tag` tokens.
- Punctuation count: characters in `string.punctuation`.
- Words: lower-cased alphanumeric tokens after URL removal.
- Sentences: split on `.`, `!`, `?` and newlines.

Sentiment path (`clean_for_sentiment`):

1. emoji → textual name (`emoji.demojize`, underscores → spaces);
2. remove URLs and @mentions, drop `#` symbols;
3. lower-case, strip non-alphanumerics and normalise whitespace;
4. remove English stop words (NLTK corpus if present locally, otherwise scikit-learn's list — no network download at runtime).

Null description → literal `"missing"` for text processing; `description_length` stays 0; `description_binary = 0`.

## 3. Feature definitions (paper Table 4 — formulas are ours)

`n_tweets` = number of tweets provided; else `tweets_observed`; else `statuses_count`.

| Feature | Definition |
|---|---|
| verified | 1 if verified badge |
| friends_count / followers_count / listed_count / favorites_count | profile counts (`favourites_count` alias accepted) |
| hashtag_count / mentions_count / url_count | Σ over observed tweets (`num_hashtags`, `num_mentions`, `num_urls` if present; otherwise counted from text) |
| retweet_count / reply_count | Σ `retweet_count` / `reply_count` over observed tweets (engagement received) |
| statuses_count | profile status count |
| ffratio | `followers_count / (friends_count + 1)` |
| avg_hashtag, avg_retweets, avg_replies, avg_mentions, avg_url | corresponding count ÷ `n_tweets` |
| avg_user_engagement | `(retweet_count + reply_count + favorites_received) / n_tweets` |
| unique_word_count | distinct word tokens across observed tweets |
| unique_word_use | `unique_word_count / total_words` |
| punctuation_count | Σ punctuation characters |
| avg_sentence_length | `total_words / sentence_count` |
| punctuation_density | `punctuation_count / total_characters` |
| profile_completeness | fraction filled of {name, description, location, url, non-default picture, banner} |
| description_binary | 1 if description non-empty |
| default_profile, default_profile_image, geo_enabled, profile_background_tile | profile flags |
| avg_polarity, avg_subjectivity | mean TextBlob polarity/subjectivity over cleaned tweets (+ description) |

Auxiliary (computed, stored, not in the 31-vector): `avg_favorites` (= favorites_received / n_tweets; appears in the paper's figures but not Table 4), `description_length`, `n_tweets`.

If no tweets are given, linguistic and sentiment features are computed from the description so that profile-only inputs are not trivially zero.

## 4. Training (paper §III, §IV)

- Shuffle + stratified hold-out split (default 75/25; 70/30 selectable).
- Optional SHAP feature selection: fit a screening model on the training split, rank by mean |SHAP|, keep top-k, retrain.
- `Pipeline(SimpleImputer(median) → MinMaxScaler → classifier)`.
- Randomised hyperparameter search (F1, stratified k-fold) with compact per-model spaces (paper publishes no grids).
- Stratified k-fold cross-validation of the tuned pipeline on the training split (per-fold accuracy, precision, recall, F1, ROC-AUC, mean ± std).
- Hold-out evaluation: metrics, confusion matrix, ROC and PR curves, probability histogram.
- Artefacts (`pipeline.joblib`, `feature_metadata.json`, `metrics.json`, `shap_global.json`, SHAP background and LIME samples) are written with SHA-256 checksums, uploaded to the organization's storage prefix and registered in the `ml_models` table; loading verifies the checksums first.

## 5. Explainability

SHAP (`ShapExplainer`):

- Tree models (RF, Extra Trees, DT, XGBoost, LightGBM): `shap.TreeExplainer`. scikit-learn forests explain class probability natively. XGBoost/LightGBM are asked for `model_output="probability"` with an interventional background; if the installed shap/booster combination refuses (observed for XGBoost 3.x + shap 0.52), the explainer falls back to raw log-odds and reports `output_scale = "log_odds"`.
- Other models: `shap.KernelExplainer` on `predict_proba[:, 1]` with a k-means-summarised background (probability scale).
- Global: mean |SHAP| ranking over a training sample (default 300 rows; 120 for Kernel SHAP), top-20 beeswarm sample, group totals.
- Local: per-feature contribution, direction (BOT/HUMAN), cumulative path (waterfall). Additivity holds: `base_value + Σ shap = model_output`.

LIME (`LimeExplainer`): `LimeTabularExplainer` fitted on the scaled training sample (up to 2,000 rows), `discretize_continuous=True`, 3,000 perturbations, explaining class BOT. Output: prediction probabilities, rules (e.g. `0.00 < ffratio <= 0.01`), weights, bot/human indicator lists, surrogate R².

## 6. Risk score

`risk_score = round(100 × P(bot))`, bands: ≥80 critical, ≥60 high, ≥40 medium, ≥20 low, else minimal. It is a presentation of the model probability, not a verdict.

## 7. Engineering adaptations (explicit)

1. Derived-feature formulas (Section 3) are ours — the paper does not define them.
2. Hyperparameter search spaces are ours — the paper only states that hyperparameters were optimised by cross-validation.
3. TextBlob is used for sentiment; the paper names the output features but not the library.
4. Cresci import computes text-derived features on up to `BOTSHIELD_MAX_TWEETS_PER_USER` (default 100) most recent tweets per account for tractability; count features use all tweets.
5. Kernel SHAP global importance is computed on a capped sample (120 rows, 200 coalitions) for non-tree models.
6. Paper-reported numbers live in `ml/paper_results.py` as reference data and are rendered only under "Reported in base paper".
7. The production application contains no synthetic, sample or seeded data; the test suite generates its own throw-away data under `backend/tests/fixtures` and `scripts/check_no_demo_data.py` enforces the rule on production code.
8. Features that are constant in the training data are dropped before fitting and listed in `feature_metadata.json` (`dropped_constant_features`). With the public user-level Cresci mirror the 11 tweet-derived features are constant (no `tweets.csv`), so production models use 20 of the 31 features; adding the authors' tweet files and re-importing restores all 31.
9. The benchmark data available for import is the public user-level mirror of Cresci-2015/2017 (`scripts/fetch_datasets.py`, provenance + SHA-256 in `backend/data/datasets/PROVENANCE.md`). Cresci-17 uses the paper's Table 3 composition (genuine, social spambots 1-3, traditional spambots #1, fake followers); the mirror's traditional_spambots_2-4 subsets are stored separately and not used.
