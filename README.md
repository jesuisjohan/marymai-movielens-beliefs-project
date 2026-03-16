# MovieLens Beliefs – Belief Rating Prediction

A Google Colab notebook project studying **belief ratings** (expected ratings for unseen movies) using the [MovieLens Beliefs Dataset](https://grouplens.org/datasets/movielens/).

---

## Research Questions

### RQ1 – Belief Gap by Genre
> *Do users tend to expect higher or lower ratings than actual community ratings, and how does this vary by genre?*

**Method:** Compute `belief_gap = userPredictRating − movie_mean_actual_rating` per genre. Bar chart with red (underestimate) / green (overestimate) coloring.

### RQ2 – Belief Formation Decomposition
> *What drives belief ratings – personal tendency (user bias), movie reputation (item popularity), or complex user-item interactions (latent factors)?*

**Method:** Incremental RMSE decomposition across 5 levels:

| Level | Model | Component |
|-------|-------|-----------|
| 0 | Global Mean | Baseline – no personalization |
| 1 | + User Bias | Personal tendency |
| 2 | + Item Bias (Additive) | Movie reputation |
| 3 | BiasedMF (SGD) | Latent user-item interactions |
| 4 | SVD++ | Implicit feedback from rating history |

### RQ3 – Belief as Predictor of Actual Rating
> *Does belief rating add value as a supplementary feature to predict actual ratings? Does combining belief + collaborative filtering outperform pure collaborative filtering?*

**Method:** Find matched (user, movie) pairs where belief was recorded **before** the actual rating. Compare Ridge / Random Forest with and without `belief_rating` as a feature.

---

## Dataset

Expected CSV files (place them all under the same directory, e.g. `data/`):

| File | Description |
|------|-------------|
| `belief_data.csv` | One row per (user, movie) elicitation. Key columns: `userId`, `movieId`, `isSeen`, `userPredictRating`, `userCertainty`, `systemPredictRating`, `tstamp` |
| `user_rating_history.csv` | Historical explicit ratings. Key columns: `userId`, `movieId`, `rating`, `timestamp` |
| `movies.csv` | Movie metadata. Key columns: `movieId`, `title`, `genres` |
| `user_recommendation_history.csv` | Recommendation log (optional). Key columns: `userId`, `movieId`, `predictedRating`, `tstamp` |
| `movie_elicitation_set.csv` | Elicitation set metadata (optional). Key columns: `movieId`, `month_idx`, `source` |

---

## Project Structure

```
.
├── notebooks/
│   └── belief_prediction.ipynb   # Main Colab notebook (all sections)
├── requirements.txt               # Python dependencies
└── README.md
```

### Notebook Structure

| Section | Content |
|---------|---------|
| 0 – Setup & Load Data | Imports, `DATA_DIR` config, load CSV files |
| 1 – EDA & Visualization | Distributions, genre chart, heatmap, rating dist |
| 2 – Preprocessing | Genre OHE, user/movie stats, time-based split |
| 3 – RQ1: Belief Gap | Belief gap by genre and certainty level |
| 4 – RQ2: Decomposition | GlobalMean → UserBias → ItemBias → BiasedMF → SVD++ + grid search |
| 5 – RQ3: Belief → Actual | Matched pairs, Ridge/RF ±belief, feature importance |
| 6 – Ablation | `systemPredictRating` leakage demo |
| 7 – Conclusion | Answers to RQ1/RQ2/RQ3, limitations, future work |

---

## Models and Their Roles

| Model | Section | Purpose |
|-------|---------|---------|
| Global Mean | RQ2 Level 0 | Baseline anchor |
| User Mean / User Bias | RQ2 Level 1 | Test if personal tendency matters |
| Movie Mean / Item Bias | RQ2 Level 2 | Test if movie reputation matters |
| Additive Bias (μ + b_u + b_i) | RQ2 Level 2 | Combined bias bridge |
| **BiasedMF (SGD)** | RQ2 Level 3 | Latent user-item interactions |
| **SVD++** | RQ2 Level 4 | Adds implicit feedback from rating history |
| Ridge Regression | RQ3 | Feature-based model ±belief feature |
| Random Forest | RQ3 | Non-linear ±belief + feature importance |
| Ridge + systemPredictRating | Section 6 | Leakage demo only |

---

## How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or clone this repository.
3. Open `notebooks/belief_prediction.ipynb`.
4. Edit the `DATA_DIR` variable in **Section 0** to point to your data folder:
   - **Google Drive**: mount your Drive and set `DATA_DIR = "/content/drive/MyDrive/your_data_folder"`.
   - **Local upload**: use Colab's file upload and set `DATA_DIR = "/content"`.
5. Run all cells (`Runtime → Run all`).

> If the data files are not present, the notebook will print a friendly guidance message and skip the computation cells gracefully — it will **not** crash on import.

---

## Anti-Leakage Principles

- **`systemPredictRating`** (and `predictedRating` from recommendation logs) are outputs of the dataset's internal recommender. Using them as features constitutes **policy leakage**. All RQ1/RQ2/RQ3 models exclude these columns.
- They are only used in **Section 6 (Ablation)** to demonstrate the leakage effect.
- **RQ3 temporal validity**: only matched (user, movie) pairs where `belief.tstamp < rating.timestamp` are included, ensuring belief predicts *future* actual ratings.

## Time-Based Split

Random splits inflate offline metrics because temporal autocorrelations in user behaviour allow the model to "see the future". This notebook uses a **per-user chronological split** on `tstamp`:

- **Train**: earliest 70 % of each user's belief rows  
- **Validation**: next 15 %  
- **Test**: latest 15 %

This mirrors real deployment: a model trained on historical data is evaluated on future elicitations.

---

## Key Implementation Features

- **Vectorized `BiasedMF._predict_batch()`**: NumPy-vectorized, no Python for-loops
- **Vectorized `predict_additive_bias()`**: pandas `.map()` + vectorized arithmetic
- **Early stopping**: Both `BiasedMF` and `SVD++` track val RMSE and stop if no improvement for `patience=5` epochs, restoring best parameters
- **SVD++ implicit feedback**: `N(u)` built from `user_rating_history.csv`
- **Hyperparameter grid search**: 27 configurations for `BiasedMF` (n_factors × lr × reg)

---

## Dependencies

```
pip install -r requirements.txt
```

Core: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

---

## Limitations

- **Selection bias**: only users who participated in the elicitation study are included.
- **MNAR (Missing Not At Random)**: `isSeen = -1` rows are non-responses; they are excluded from the belief prediction task (not treated as negatives).
- **Leakage**: `systemPredictRating` is excluded from RQ1/RQ2/RQ3 models; discussed in the ablation section.
- **RQ3 matched pairs**: The intersection of belief data and actual rating history may be small; a warning is printed if fewer than 100 matched pairs are found.
- **Cold-start**: limited item content features (title + genres only); no external metadata used.
