# MovieLens Beliefs – Belief Rating Prediction

A Google Colab notebook project for predicting **belief ratings** (expected ratings for unseen movies) using the [MovieLens Beliefs Dataset](https://grouplens.org/datasets/movielens/).

---

## Research Question

> **Can a user's past rating history and movie genre information reliably predict the user's expected rating (belief) for a movie they have not yet seen, and does user-reported certainty correlate with prediction error?**

---

## Dataset

Expected CSV files (place them all under the same directory, e.g. `data/`):

| File | Description |
|------|-------------|
| `belief_data.csv` | One row per (user, movie) elicitation. Key columns: `userId`, `movieId`, `isSeen`, `userPredictRating`, `userCertainty`, `systemPredictRating`, `tstamp` |
| `user_rating_history.csv` | Historical explicit ratings. Key columns: `userId`, `movieId`, `rating`, `timestamp` |
| `movies.csv` | Movie metadata. Key columns: `movieId`, `title`, `genres` |
| `user_recommendation_history.csv` | Recommendation log. Key columns: `userId`, `movieId`, `predictedRating`, `tstamp` |
| `movie_elicitation_set.csv` | Elicitation set metadata. Key columns: `movieId`, `month_idx`, `source` |

---

## Project Structure

```
.
├── notebooks/
│   └── belief_prediction.ipynb   # Main Colab notebook (all sections)
├── requirements.txt               # Python dependencies
└── README.md
```

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

- **`systemPredictRating`** (and `predictedRating` from recommendation logs) are outputs of the dataset's internal recommender. Using them as features in the main model constitutes **policy leakage**. The main model (`BiasedMF`) does **not** use these columns.
- They are only used in **Section 6 (Ablation)** to demonstrate the leakage effect and provide an upper-bound comparison.

## Time-Based Split

Random splits inflate offline metrics because temporal autocorrelations in user behaviour allow the model to "see the future". This notebook uses a **per-user chronological split** on `tstamp`:

- **Train**: earliest 70 % of each user's belief rows  
- **Validation**: next 15 %  
- **Test**: latest 15 %

This mirrors real deployment: a model trained on historical data is evaluated on future elicitations.

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
- **Leakage**: `systemPredictRating` is excluded from the main model; discussed in the ablation section.
- **Cold-start**: limited item content features (title + genres only); no external metadata used.
