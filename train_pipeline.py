"""
Mental Health Risk Prediction - Model Training Pipeline
----------------------------------------------------------------
Rebuilds the same preprocessing / modeling approach used in the original
research notebook (cleaning -> feature engineering -> encoding ->
feature selection -> scaling -> SMOTE-style balancing -> model
comparison -> tuning -> final model selection), applied to the actual
survey dataset supplied for this project.

Target definition
------------------
The source data does not ship a ready-made binary risk label. It does
carry a clinician-style `severity` rating (Mild / Moderate / Severe).
We treat "Mild" as Low Risk (0) and "Moderate"/"Severe" as High Risk (1) -
this is the natural business framing (a wellbeing screening flag) and
keeps the split reasonably balanced (242 / 258 respondents).

`mental_health_condition`, `severity` and `treatment` are all downstream
of the same clinical assessment that produced the target, so they are
dropped from the feature set to avoid leakage. Only the behavioural /
survey-response columns are used as predictors, exactly the kind of
input a new user would supply on the prediction form.
"""

import pickle
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("dataset/mental_health_prediction.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - df.shape[0]} duplicate rows")

numeric_cols = [
    "sleep_hours", "sleep_quality", "social_media_hours",
    "academic_work_pressure", "physical_activity_days", "stress_level",
    "anxiety_score", "depression_score", "work_life_balance", "mood_score",
    "concentration_level", "social_support",
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["gender"] = df["gender"].str.strip()
df["occupation"] = df["occupation"].str.strip()

print("Remaining missing values:", df.isnull().sum().sum())

df["data_source"] = "real"
df["respondent_id"] = df.index  # each real respondent gets a stable id

# ---------------------------------------------------------------------
# 2b. Synthetic data generation
# ---------------------------------------------------------------------
# The real survey only has 500 respondents. To give the model more
# training signal (and to demonstrate a realistic augmentation
# technique), we generate synthetic respondents by bootstrapping a real
# row and perturbing its numeric answers with small, bounded noise.
# Categorical/clinical fields (gender, occupation, condition, severity,
# treatment) are copied as-is from the sampled row, since those are
# what the numeric answers were actually rated against - jittering the
# numbers slightly around a real profile keeps the label meaningful
# without inventing new clinical categories.
INTEGER_SCALE_COLS = [
    "sleep_quality", "academic_work_pressure", "physical_activity_days",
    "stress_level", "anxiety_score", "depression_score", "work_life_balance",
    "mood_score", "concentration_level", "social_support",
]
CONTINUOUS_COLS = ["sleep_hours", "social_media_hours"]

SYNTHETIC_MULTIPLIER = 3  # generates 3 synthetic rows per real row
NOISE_FRACTION = 0.08     # jitter size, as a fraction of each column's std


def generate_synthetic_rows(source_df: pd.DataFrame, multiplier: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    col_bounds = {
        col: (source_df[col].min(), source_df[col].max())
        for col in INTEGER_SCALE_COLS + CONTINUOUS_COLS + ["age"]
    }
    col_std = {col: max(source_df[col].std(), 1e-6) for col in INTEGER_SCALE_COLS + CONTINUOUS_COLS + ["age"]}

    n_synthetic = len(source_df) * multiplier
    sampled_idx = rng.choice(source_df.index, size=n_synthetic, replace=True)
    synthetic = source_df.loc[sampled_idx].reset_index(drop=True).copy()

    for col in INTEGER_SCALE_COLS:
        noise = rng.normal(0, col_std[col] * NOISE_FRACTION, size=n_synthetic)
        low, high = col_bounds[col]
        synthetic[col] = np.clip(np.round(synthetic[col].values + noise), low, high)

    for col in CONTINUOUS_COLS:
        noise = rng.normal(0, col_std[col] * NOISE_FRACTION, size=n_synthetic)
        low, high = col_bounds[col]
        synthetic[col] = np.clip(np.round(synthetic[col].values + noise, 1), low, high)

    age_noise = rng.randint(-1, 2, size=n_synthetic)  # -1, 0, or +1 year
    low, high = col_bounds["age"]
    synthetic["age"] = np.clip(synthetic["age"].values + age_noise, low, high)

    synthetic["data_source"] = "synthetic"
    return synthetic


synthetic_df = generate_synthetic_rows(df, SYNTHETIC_MULTIPLIER, RANDOM_STATE)
df = pd.concat([df, synthetic_df], ignore_index=True)
print(f"Generated {len(synthetic_df)} synthetic rows -> combined dataset: {df.shape[0]} rows")

df.to_csv("dataset/mental_health_prediction_augmented.csv", index=False)
print("Saved combined (real + synthetic) dataset to dataset/mental_health_prediction_augmented.csv")

# ---------------------------------------------------------------------
# 3. Target construction
# ---------------------------------------------------------------------
df["Mental_Health_Risk"] = (df["severity"] != "Mild").astype(int)
print(df["Mental_Health_Risk"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------
# Wellbeing_Index - composite of sleep, activity and mood, discounted by
# heavy social-media use (mirrors the notebook's Lifestyle_Score idea).
df["Wellbeing_Index"] = (
    df["sleep_hours"] + df["physical_activity_days"] + df["mood_score"]
    - df["social_media_hours"] / 2
)

# Pressure_Index - academic/work pressure amplified by poor work-life
# balance (mirrors the notebook's Workload_Pressure idea).
df["Pressure_Index"] = df["academic_work_pressure"] * (10 - df["work_life_balance"]) / 10

# ---------------------------------------------------------------------
# 5. Encoding
# ---------------------------------------------------------------------
le_gender = LabelEncoder()
df["Gender_Enc"] = le_gender.fit_transform(df["gender"])

# Occupation is nominal with 3 levels -> one-hot, drop_first to avoid
# the dummy-variable trap (same treatment as Department in the notebook).
df_encoded = pd.get_dummies(df, columns=["occupation"], prefix="Occ", drop_first=True)

leak_cols = ["gender", "mental_health_condition", "severity", "treatment", "data_source"]
df_encoded = df_encoded.drop(columns=leak_cols)

occ_cols = [c for c in df_encoded.columns if c.startswith("Occ_")]
print(f"Shape after encoding: {df_encoded.shape}")

# ---------------------------------------------------------------------
# 6. Feature selection (drop weak-correlation features, |corr| < 0.03)
# ---------------------------------------------------------------------
feature_corr = (
    df_encoded.corr(numeric_only=True)["Mental_Health_Risk"]
    .drop(["Mental_Health_Risk", "respondent_id"])
    .sort_values(key=abs, ascending=False)
)
weak_features = feature_corr[feature_corr.abs() < 0.03].index.tolist()
print(f"Dropping {len(weak_features)} weak features: {weak_features}")
model_df = df_encoded.drop(columns=weak_features)
print(f"Final modeling dataset shape: {model_df.shape}")

# ---------------------------------------------------------------------
# 7. Train / test split
# ---------------------------------------------------------------------
# Synthetic rows are perturbed copies of a real respondent, so a naive
# random split could put a respondent's real row in training and its
# near-identical synthetic siblings in test (or vice versa) - inflating
# the reported score. StratifiedGroupKFold keeps every row that shares
# a respondent_id on the same side of the split while still balancing
# the Low Risk / High Risk classes across the split.
X = model_df.drop(columns=["Mental_Health_Risk", "respondent_id"])
y = model_df["Mental_Health_Risk"]
groups = model_df["respondent_id"]

from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print(f"Train: {X_train.shape}, Test: {X_test.shape}, respondent overlap between splits: {len(overlap)}")

# ---------------------------------------------------------------------
# 8. Scaling
# ---------------------------------------------------------------------
scale_cols = [c for c in numeric_cols + ["age", "Wellbeing_Index", "Pressure_Index"]
              if c in X_train.columns]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------------------------------------------------------------------
# 9. Balance training data (custom SMOTE-style oversampling)
# ---------------------------------------------------------------------
def smote_oversample(X_minority, n_samples, k=5, random_state=RANDOM_STATE):
    """Generate synthetic minority-class samples by interpolating between
    each point and one of its k nearest minority-class neighbors."""
    rng = np.random.RandomState(random_state)
    k = min(k, len(X_minority) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, indices = nn.kneighbors(X_minority)

    synthetic = []
    for _ in range(n_samples):
        i = rng.randint(0, X_minority.shape[0])
        neighbor_idx = indices[i, rng.randint(1, k + 1)]
        gap = rng.rand()
        new_point = X_minority[i] + gap * (X_minority[neighbor_idx] - X_minority[i])
        synthetic.append(new_point)
    return np.array(synthetic)


train_counts = y_train.value_counts()
minority_class, majority_class = train_counts.idxmin(), train_counts.idxmax()
n_to_generate = int(train_counts[majority_class] - train_counts[minority_class])

X_train_arr = X_train_scaled.values
minority_mask = (y_train == minority_class).values

if n_to_generate > 0:
    synthetic_X = smote_oversample(X_train_arr[minority_mask], n_to_generate)
    synthetic_y = np.full(n_to_generate, minority_class)
    X_train_bal = np.vstack([X_train_arr, synthetic_X])
    y_train_bal = np.concatenate([y_train.values, synthetic_y])
else:
    X_train_bal, y_train_bal = X_train_arr, y_train.values

print("Before balancing:", dict(train_counts))
print("After balancing:", dict(pd.Series(y_train_bal).value_counts()))

# ---------------------------------------------------------------------
# 10. Model comparison
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = []
fitted_models = {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1 Score": f1_score(y_test, preds, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, proba),
    })
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
print(results_df.round(4))

# ---------------------------------------------------------------------
# 11. Hyperparameter tuning on Random Forest
# ---------------------------------------------------------------------
param_grid = {
    "n_estimators": [150, 250],
    "max_depth": [10, 14, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid, scoring="roc_auc", cv=3, n_jobs=-1,
)
grid_search.fit(X_train_bal, y_train_bal)
tuned_model = grid_search.best_estimator_
tuned_preds = tuned_model.predict(X_test_scaled)
tuned_proba = tuned_model.predict_proba(X_test_scaled)[:, 1]
tuned_auc = roc_auc_score(y_test, tuned_proba)
print("Best params:", grid_search.best_params_, "Tuned test AUC:", round(tuned_auc, 4))

# ---------------------------------------------------------------------
# 12. Final model selection
# ---------------------------------------------------------------------
rf_untuned_auc = results_df.loc[results_df["Model"] == "Random Forest", "ROC AUC"].values[0]
if tuned_auc > rf_untuned_auc + 0.01:
    final_model = tuned_model
    final_model_name = "Random Forest (Tuned)"
else:
    final_model = fitted_models["Random Forest"]
    final_model_name = "Random Forest"

final_preds = final_model.predict(X_test_scaled)
final_proba = final_model.predict_proba(X_test_scaled)[:, 1]
print(f"\nFinal model: {final_model_name}")
print("Test Accuracy :", round(accuracy_score(y_test, final_preds), 4))
print("Test ROC AUC  :", round(roc_auc_score(y_test, final_proba), 4))
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds, target_names=["Low Risk", "High Risk"]))

importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(10).round(4))

# ---------------------------------------------------------------------
# 13. Persist model + preprocessing bundle
# ---------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(final_model, f)

preprocessing_bundle = {
    "scaler": scaler,
    "scale_cols": scale_cols,
    "le_gender": le_gender,
    "occ_cols": occ_cols,
    "feature_names": X.columns.tolist(),
    "weak_features_dropped": weak_features,
    "numeric_cols": numeric_cols,
    "gender_classes": list(le_gender.classes_),
    "occupation_classes": ["Both (Part-time work + Study)", "Student", "Working Professional"],
    "final_model_name": final_model_name,
    "test_metrics": {
        "accuracy": float(accuracy_score(y_test, final_preds)),
        "precision": float(precision_score(y_test, final_preds)),
        "recall": float(recall_score(y_test, final_preds)),
        "f1": float(f1_score(y_test, final_preds)),
        "roc_auc": float(roc_auc_score(y_test, final_proba)),
    },
}
with open("model/preprocessing.pkl", "wb") as f:
    pickle.dump(preprocessing_bundle, f)

print("\nSaved model.pkl and preprocessing.pkl")

# ---------------------------------------------------------------------
# 14. Analysis plots + metrics export (for the app's Analytics page)
# ---------------------------------------------------------------------
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve

PLOTS_DIR = "static/images/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "ink": "#202a33", "dusk": "#24344d", "sage": "#6e9887",
    "amber": "#d6a253", "danger": "#b3563f", "paper": "#f3f1ec",
    "mist": "#c9d6d3",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["paper"], "axes.facecolor": PALETTE["paper"],
    "axes.edgecolor": "#d8d3c6", "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"], "font.size": 10, "figure.dpi": 130,
})

# 14a. Model comparison - grouped bar chart across all metrics
metrics_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(results_df))
width = 0.15
colors = [PALETTE["dusk"], PALETTE["sage"], PALETTE["amber"], PALETTE["danger"], "#7a8ba0"]
for i, col in enumerate(metrics_cols):
    ax.bar(x + i * width, results_df[col], width, label=col, color=colors[i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_title("Model comparison across evaluation metrics")
ax.legend(loc="lower right", ncol=3, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/model_comparison.png")
plt.close(fig)

# 14b. F1 score ranking - horizontal bar, sorted
fig, ax = plt.subplots(figsize=(9, 5))
f1_sorted = results_df.sort_values("F1 Score")
bar_colors = [PALETTE["amber"] if m == final_model_name.replace(" (Tuned)", "") else PALETTE["dusk"]
              for m in f1_sorted["Model"]]
ax.barh(f1_sorted["Model"], f1_sorted["F1 Score"], color=bar_colors)
for i, v in enumerate(f1_sorted["F1 Score"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_title("F1 score by algorithm (test set)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/f1_scores.png")
plt.close(fig)

# 14c. Confusion matrix - final model
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, final_preds)
im = ax.imshow(cm, cmap="Greens")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
             color="white" if v > cm.max() / 2 else PALETTE["ink"], fontsize=14, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Low Risk", "High Risk"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Low Risk", "High Risk"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — {final_model_name}")
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
plt.close(fig)

# 14d. ROC curves - all models overlaid
fig, ax = plt.subplots(figsize=(7, 6))
palette_cycle = [PALETTE["dusk"], PALETTE["sage"], PALETTE["amber"], PALETTE["danger"],
                  "#7a8ba0", "#9b6b9e", "#5a8f9e"]
for i, (name, mdl) in enumerate(fitted_models.items()):
    proba = mdl.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=palette_cycle[i % len(palette_cycle)], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="#aaa", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — all algorithms")
ax.legend(fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/roc_curves.png")
plt.close(fig)

# 14e. Feature importance - final model
fig, ax = plt.subplots(figsize=(8, 6))
top_importances = importances.head(12).sort_values()
ax.barh(top_importances.index, top_importances.values, color=PALETTE["sage"])
ax.set_title(f"Feature importance — {final_model_name}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/feature_importance.png")
plt.close(fig)

# 14f. Class balance before / after SMOTE-style oversampling
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
before_counts = pd.Series(y_train).value_counts().sort_index()
after_counts = pd.Series(y_train_bal).value_counts().sort_index()
labels = ["Low Risk", "High Risk"]
axes[0].bar(labels, before_counts.values, color=[PALETTE["sage"], PALETTE["danger"]])
axes[0].set_title("Training set — before balancing")
axes[1].bar(labels, after_counts.values, color=[PALETTE["sage"], PALETTE["danger"]])
axes[1].set_title("Training set — after balancing")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/class_balance.png")
plt.close(fig)

# 14g. Correlation heatmap of modeling features vs target
fig, ax = plt.subplots(figsize=(7, 8))
corr_all = model_df.corr(numeric_only=True)["Mental_Health_Risk"].drop(["Mental_Health_Risk", "respondent_id"]).sort_values()
bar_colors2 = [PALETTE["danger"] if v < 0 else PALETTE["sage"] for v in corr_all.values]
ax.barh(corr_all.index, corr_all.values, color=bar_colors2)
ax.axvline(0, color="#888", linewidth=0.8)
ax.set_title("Feature correlation with Mental_Health_Risk")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/correlation.png")
plt.close(fig)

print(f"\nSaved 7 analysis plots to {PLOTS_DIR}/")

# 14i. Dataset composition - real vs synthetic
fig, ax = plt.subplots(figsize=(6, 4.5))
comp_counts = df["data_source"].value_counts()
ax.bar(comp_counts.index, comp_counts.values, color=[PALETTE["dusk"], PALETTE["amber"]])
for i, v in enumerate(comp_counts.values):
    ax.text(i, v + max(comp_counts.values) * 0.01, str(v), ha="center", fontsize=10)
ax.set_title("Dataset composition — real vs. synthetic rows")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/dataset_composition.png")
plt.close(fig)
print("Saved dataset_composition.png")

# 14h. Export metrics table (incl. F1) as JSON for the Django Analytics page
analytics_export = {
    "final_model_name": final_model_name,
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_shape": {
        "rows": int(df.shape[0]),
        "real_rows": int((df["data_source"] == "real").sum()),
        "synthetic_rows": int((df["data_source"] == "synthetic").sum()),
        "features_used": int(X.shape[1]),
    },
    "class_balance": {
        "before": {"Low Risk": int(before_counts.get(0, 0)), "High Risk": int(before_counts.get(1, 0))},
        "after": {"Low Risk": int(after_counts.get(0, 0)), "High Risk": int(after_counts.get(1, 0))},
    },
    "model_results": results_df.rename(
        columns={"F1 Score": "F1_Score", "ROC AUC": "ROC_AUC"}
    ).round(4).to_dict(orient="records"),
    "best_params": grid_search.best_params_,
    "final_test_metrics": preprocessing_bundle["test_metrics"],
    "top_features": [{"feature": k, "importance": round(float(v), 4)} for k, v in importances.head(12).items()],
    "weak_features_dropped": weak_features,
    "plots": [
        {"file": "dataset_composition.png", "title": "Dataset composition",
         "description": "Real survey respondents vs. synthetically generated rows used for training."},
        {"file": "model_comparison.png", "title": "Model comparison across metrics",
         "description": "Accuracy, precision, recall, F1 and ROC AUC for every algorithm tested."},
        {"file": "f1_scores.png", "title": "F1 score by algorithm",
         "description": "Algorithms ranked by F1 score on the held-out test set."},
        {"file": "confusion_matrix.png", "title": "Confusion matrix",
         "description": f"Predicted vs actual outcomes for the final model ({final_model_name})."},
        {"file": "roc_curves.png", "title": "ROC curves",
         "description": "True vs false positive rate trade-off for every algorithm tested."},
        {"file": "feature_importance.png", "title": "Feature importance",
         "description": f"Top contributing features for the final model ({final_model_name})."},
        {"file": "class_balance.png", "title": "Class balance before/after SMOTE",
         "description": "Training-set label distribution before and after synthetic oversampling."},
        {"file": "correlation.png", "title": "Feature correlation with target",
         "description": "Pearson correlation of each engineered feature with Mental_Health_Risk."},
    ],
}
with open("model/analytics.json", "w") as f:
    json.dump(analytics_export, f, indent=2)

print("Saved model/analytics.json")

