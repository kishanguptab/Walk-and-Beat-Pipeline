# Walk-and-Beat-Pipeline
A pace aware Music Recommendation Platform PoC
Walk & Beat — Cadence-Matched Music Recommender (POC)
Goal: build a pace-matched music recommender that keeps people walking/jogging longer by aligning song tempo (BPM) with step rate (spm) inferred from phone IMU.
This repo includes: end-to-end data → features → PCA → ML models (step-rate regressor + activity classifier) → music matching (Iteration-1) → HMM-smoothed recommendations with duration-aware song changes (Iteration-2), plus MLflow tracking and a zero-training evaluation notebook.

🔎 What’s in this repo?
notebooks/walk-beat-core.ipynb — main notebook (data ingestion → feature engineering → PCA → training → evaluation → iterations).

notebooks/eval_only.ipynb — evaluation-only pipeline (loads saved models & splits; runs Iteration-1 + Iteration-2; saves metrics).

artifacts/ — saved models & datasets for evaluation:

best_regressor.pkl, best_classifier.pkl

features_pca.parquet (optional full set for per-user recs)

eval/X_test.parquet, eval/y_reg_test.csv, eval/y_class_test.csv

data/ (not versioned) —  WISDM & Spotify CSVs here 
Data Sources
IMU: WISDM v1.1 raw accelerometer (users, activity, time, x/y/z @20 Hz).

Music: 2023 Spotify CSVs (features, tracks, artists, albums, data). We join by track id to get track_name, artist_name, bpm (from tempo) and duration_ms.


Environment
Python 3.10+

pandas numpy scikit-learn mlflow joblib matplotlib seaborn pyarrow fastparquet

(Optional) plotly for interactive PCA scatter
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn mlflow joblib matplotlib seaborn pyarrow fastparquet plotly
1) Train + Track (MLflow)
Open notebooks/walk and beat.ipynb and run all cells. You’ll get:

Best models saved to artifacts/best_regressor.pkl and artifacts/best_classifier.pkl

Eval splits saved to artifacts/eval/:

X_test.parquet, y_reg_test.csv, y_class_test.csv

MLflow runs under mlruns/

Plots saved to figures/

2) Evaluate Only 
Open notebooks/*eval.ipynb and run all cells. It does not train. It:

Loads models & eval splits from artifacts/

Builds the music catalog (song_df) from Spotify CSVs

Runs Iteration-1 (direct BPM match) + Iteration-2 (HMM + duration FSM)

Saves results:

artifacts/eval/iter1_recommendations.parquet

artifacts/eval/iter1_user_metrics.csv

artifacts/eval/iter2_hmm_transition_matrix.csv

artifacts/eval/iter2_hmm_recommendations.parquet

artifacts/eval/iter2_hmm_user_metrics.csv

If best_classifier.pkl is present, prints a classification report and confusion matrix

Recommendation Iterations
Iteration 1 — Direct: window-level cadence → pick nearest-BPM song (±10).
Great cadence fit, but too many switches (≈ one every 5 s).

Iteration 2 — HMM + Duration FSM:

Discretize cadence into states {slow_walk, fast_walk, jog}

Learn transition matrix on predicted state sequences

Viterbi smoothing (reduce jitter)

Change song only on robust state change or song end
Result: ~6.3 switches/session, ~143 s segments; moderate cadence fit—acceptable UX trade-off for POC.

Example Results (from our runs)
Step Rate Regression (Test): MAE ≈ 4.29 spm, CWA±10% ≈ 98.7%, CER>25% ≈ 0.18%

Activity Classification (Test): Accuracy ≈ 0.993, F1 (weighted) ≈ 0.993

Iteration-1: Cadence-fit ≈ 1.00, ~200 switches/session, mean segment ≈ 5 s

Iteration-2 (HMM): Cadence-fit ≈ 0.52, ~6.3 switches/session, mean segment ≈ 143 s

MLflow
Experiment name: cadence_activity_modeling

We run a tiny RF grid (n_estimators, max_depth) and log:

validation MAE (regression), validation accuracy (classification)

models as artifacts (best are also dumped to artifacts/)

Start the UI locally:

mlflow ui --backend-store-uri mlruns
