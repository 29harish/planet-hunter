
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay
)

# ---------------------------------
# 1. Project configuration
# ---------------------------------

RANDOM_STATE = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

DATASET = DATA_DIR / "kepler_cleaned.csv"

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_model_snr",
    "koi_impact",
    "koi_time0bk",
    "koi_num_transits"
]

# ---------------------------------
# 2. Load prepared data
# ---------------------------------

print("\nLoading NASA Kepler dataset...")

df = pd.read_csv(DATASET)

required = FEATURES + ["target", "kepid"]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Required columns missing: {missing}"
    )

X = df[FEATURES].copy()
y = df["target"].astype(int)
groups = df["kepid"]

# ---------------------------------
# 3. Grouped train/test split
# ---------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

assert set(groups_train).isdisjoint(
    set(groups_test)
), "Star IDs overlap between train and test!"

print("\n===== DATA SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training stars:", groups_train.nunique())
print("Testing stars:", groups_test.nunique())

print("\nTraining class counts:")
print(y_train.value_counts())

print("\nTesting class counts:")
print(y_test.value_counts())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both classes are required in both splits. "
        "Try changing RANDOM_STATE."
    )

# ---------------------------------
# 4. Define ML models
# ---------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("classifier", GradientBoostingClassifier(
            n_estimators=120,
            random_state=RANDOM_STATE
        ))
    ])
}

# ---------------------------------
# 5. Select the model using
# grouped cross-validation on
# training data only
# ---------------------------------

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []

print("\n===== CROSS-VALIDATION =====")

for name, model in models.items():

    print(f"\nTraining: {name}")

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    mean_score = scores.mean()

    print("Fold F1-scores:", scores.round(4))
    print("Mean F1-score:", round(mean_score, 4))

    cv_results.append({
        "Model": name,
        "CV_F1_Mean": mean_score,
        "CV_F1_Std": scores.std()
    })

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values(
    "CV_F1_Mean",
    ascending=False
)

cv_df.to_csv(
    REPORT_DIR / "cross_validation_results.csv",
    index=False
)

best_name = cv_df.iloc[0]["Model"]

print("\nBest model selected:", best_name)

# ---------------------------------
# 6. Fit each model on training data
# and evaluate on held-out data
# ---------------------------------

results = []
fitted_models = {}

print("\n===== TEST RESULTS =====")

for name, model in models.items():

    model.fit(X_train, y_train)

    fitted_models[name] = model

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test, probabilities
        )
    }

    results.append(metrics)

    print(f"\n{name}")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "False Positive",
            "Candidate / Confirmed"
        ],
        zero_division=0
    ))

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

# ---------------------------------
# 7. Save selected model
# ---------------------------------

best_model = fitted_models[best_name]

joblib.dump(
    best_model,
    MODEL_DIR / "planet_model.pkl"
)

joblib.dump(
    FEATURES,
    MODEL_DIR / "feature_names.pkl"
)

print("\nSaved best model:", best_name)

# ---------------------------------
# 8. Confusion matrix
# ---------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "False Positive",
        "Candidate"
    ]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(
    f"Confusion Matrix: {best_name}"
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200
)
plt.close()

# ---------------------------------
# 9. ROC curve
# ---------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    ax=ax,
    name=best_name
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)

ax.set_title("ROC Curve")

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "roc_curve.png",
    dpi=200
)
plt.close()

# ---------------------------------
# 10. Model comparison chart
# ---------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1_Score"
)

plt.ylim(0, 1)
plt.title("Test F1-Score by Model")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "model_comparison.png",
    dpi=200
)
plt.close()

# ---------------------------------
# 11. Save evaluation predictions
# ---------------------------------

prediction_df = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": best_predictions,
    "Candidate_Probability":
        best_model.predict_proba(X_test)[:, 1]
})

prediction_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)

print("\nAll results saved!")
print("Model: models/planet_model.pkl")
print("Reports: reports/")

print("\nSTEP 4 COMPLETED SUCCESSFULLY!")
