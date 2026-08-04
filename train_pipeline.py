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
df = pd.read_csv("mental_health_prediction.csv")
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

leak_cols = ["gender", "mental_health_condition", "severity", "treatment"]
df_encoded = df_encoded.drop(columns=leak_cols)

occ_cols = [c for c in df_encoded.columns if c.startswith("Occ_")]
print(f"Shape after encoding: {df_encoded.shape}")

# ---------------------------------------------------------------------
# 6. Feature selection (drop weak-correlation features, |corr| < 0.03)
# ---------------------------------------------------------------------
feature_corr = (
    df_encoded.corr(numeric_only=True)["Mental_Health_Risk"]
    .drop("Mental_Health_Risk")
    .sort_values(key=abs, ascending=False)
)
weak_features = feature_corr[feature_corr.abs() < 0.03].index.tolist()
print(f"Dropping {len(weak_features)} weak features: {weak_features}")
model_df = df_encoded.drop(columns=weak_features)
print(f"Final modeling dataset shape: {model_df.shape}")

# ---------------------------------------------------------------------
# 7. Train / test split
# ---------------------------------------------------------------------
X = model_df.drop(columns=["Mental_Health_Risk"])
y = model_df["Mental_Health_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

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
with open("model.pkl", "wb") as f:
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
with open("preprocessing.pkl", "wb") as f:
    pickle.dump(preprocessing_bundle, f)

print("\nSaved model.pkl and preprocessing.pkl")
