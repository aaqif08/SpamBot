# Dataset provenance

## Cresci-2015 / Cresci-2017 (user level)

- Source mirror: https://raw.githubusercontent.com/dblop/bot_detector_project/HEAD/users_cresci.csv
- SHA-256 of downloaded mirror: `799a5654b205b590c7675ee2d7023359410fd199a5c3d283fd3ab7a9840c889a`
- Original authors / citation:
  - S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, M. Tesconi, "Fame for sale: Efficient detection of fake Twitter followers", Decision Support Systems 80 (2015). [Cresci-2015]
  - S. Cresci et al., "The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race", WWW Companion (2017). [Cresci-2017]
- Content: account profiles (`users.csv`) only. `tweets.csv` files are distributed by the authors on request (MIB project) and are not included; tweet-derived features are therefore unavailable until they are added next to each `users.csv`.
- Terms: academic/research use as set by the original authors.

| Dataset | Subset | Accounts |
|---|---|---|
| cresci-15 | TFP | 469 |
| cresci-15 | E13 | 1,481 |
| cresci-15 | FSF | 1,169 |
| cresci-15 | INT | 1,337 |
| cresci-15 | TWT | 845 |
| cresci-17 | genuine_accounts | 3,474 |
| cresci-17 | social_spambots_1 | 991 |
| cresci-17 | social_spambots_2 | 3,457 |
| cresci-17 | social_spambots_3 | 464 |
| cresci-17 | traditional_spambots_1 | 1,000 |
| cresci-17 | fake_followers | 3,351 |
| cresci-17-extra (not in paper Table 3) | traditional_spambots_2 | 100 |
| cresci-17-extra (not in paper Table 3) | traditional_spambots_3 | 403 |
| cresci-17-extra (not in paper Table 3) | traditional_spambots_4 | 1,128 |

## Twitter Human Bots (external)

- Source: https://huggingface.co/datasets/airt-ml/twitter-human-bots/resolve/main/twitter_human_bots_dataset.csv (Hugging Face `airt-ml/twitter-human-bots`, CC BY-SA 3.0)
- SHA-256: `b9d97efe538751d8d990cd61cec1661c7cc75bb0a4b1f13dc9ff88115f900a94`
- 37,438 real accounts, profile level; labels bot/human. `listed_count` and `profile_background_tile` are not provided by this dataset and are filled with 0.
