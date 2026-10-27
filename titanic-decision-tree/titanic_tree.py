"""Train and evaluate a Titanic survival decision tree.

Use Kaggle's train.csv by placing it at data/train.csv, or omit it and the
script downloads the public seaborn Titanic CSV.
"""
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
KAGGLE_FILE = DATA_DIR / "train.csv"
SEABORN_FILE = DATA_DIR / "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
FEATURES = ["age", "sex", "pclass", "fare"]
TARGET = "survived"


def load_data() -> tuple[pd.DataFrame, str]:
    """Load Kaggle's training data if present, otherwise the public CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    if KAGGLE_FILE.exists():
        data = pd.read_csv(KAGGLE_FILE)
        source = "Kaggle train.csv"
    else:
        if not SEABORN_FILE.exists():
            print(f"Downloading public Titanic dataset from {DATA_URL}")
            with urlopen(DATA_URL, timeout=30) as response:
                SEABORN_FILE.write_bytes(response.read())
        data = pd.read_csv(SEABORN_FILE)
        source = "public seaborn Titanic CSV"
    data.columns = [str(column).lower() for column in data.columns]
    return data, source


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df, source = load_data()

    # Keep only the requested predictors and the known survival label.
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    print("TITANIC DATA EXPLORATION")
    print(f"Source: {source}")
    print(f"Rows and columns: {df.shape}")
    print("Missing values before preprocessing:")
    print(df.isna().sum().to_string())
    print("\nSurvival counts (0 = did not survive, 1 = survived):")
    print(df[TARGET].value_counts().sort_index().to_string())

    X = df[FEATURES]
    y = df[TARGET]
    numeric = ["age", "pclass", "fare"]
    categorical = ["sex"]

    # Fit imputers and encoder only on training folds to prevent leakage.
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\nMODEL EVALUATION")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Decision tree accuracy: {accuracy:.3f} ({accuracy:.1%})")
    print("Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(matrix)
    print("Classification report:")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))

    prep = model.named_steps["preprocess"]
    names = prep.get_feature_names_out()
    importances = model.named_steps["tree"].feature_importances_
    importance = pd.DataFrame({"feature": names, "importance": importances})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    print("Top model features:")
    print(importance.head(10).to_string(index=False, formatters={"importance": "{:.3f}".format}))

    top = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(top["feature"].str.replace("numeric__", "", regex=False)
                   .str.replace("categorical__", "", regex=False), top["importance"],
                   color="#2463a6")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_title("Titanic Decision Tree: Top Feature Importances", loc="left", weight="bold")
    ax.set_xlabel("Importance (normalized; higher means more influence in this fitted tree)")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (RESULTS_DIR / "evaluation.txt").open("w", encoding="utf-8") as f:
        f.write(f"Source: {source}\nRows: {len(df)}\nTest rows: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.3f} ({accuracy:.1%})\n")
        f.write("Confusion matrix (actual rows [0,1], predicted columns [0,1]):\n")
        f.write(str(matrix) + "\n\n")
        f.write(classification_report(y_test, predictions, digits=3, zero_division=0))
    top_name = str(importance.iloc[0]["feature"]).replace("numeric__", "").replace("categorical__", "")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    ax.text(0.02, 0.91, "MODEL EVALUATION", fontsize=16, weight="bold", color="#17324d")
    ax.text(0.02, 0.70, f"Holdout accuracy: {accuracy:.1%}", fontsize=22, weight="bold", color="#2463a6")
    ax.text(0.02, 0.55, f"Test passengers: {len(X_test)}     Training passengers: {len(X_train)}", fontsize=11)
    ax.text(0.02, 0.37, "Confusion matrix", fontsize=12, weight="bold")
    ax.text(0.02, 0.19, f"Actual 0 -> predicted 0: {matrix[0, 0]}     Actual 0 -> predicted 1: {matrix[0, 1]}\n"
            f"Actual 1 -> predicted 0: {matrix[1, 0]}     Actual 1 -> predicted 1: {matrix[1, 1]}",
            fontsize=10, family="monospace", linespacing=1.6)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "evaluation_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = f"""# Titanic Survival Decision Tree Report

## Objective
Use age, sex, passenger class, and fare to predict Titanic passenger survival.

## Steps
1. Loaded {len(df)} rows from {source} with Pandas and inspected dimensions, null counts, and survival counts.
2. Imputed missing age/fare values with training-set medians and missing sex with the training-set mode.
3. One-hot encoded sex and retained age, passenger class, and fare as numeric predictors.
4. Split into {len(X_train)} training and {len(X_test)} test passengers (80/20, stratified, random_state=42).
5. Fit a depth-limited scikit-learn decision tree and measured its performance on the held-out test set.

## Results
- **Accuracy:** {accuracy:.1%} ({accuracy:.3f}) on {len(X_test)} held-out passengers.
- **Top feature in this fitted tree:** `{top_name}` (importance {importance.iloc[0]['importance']:.3f}).
- **Confusion matrix:** see `evaluation.txt`; rows are actual [did not survive, survived] and columns predicted [did not survive, survived].

See `evaluation_summary.png` for a visual evaluation snapshot and `feature_importance.png` for the importance chart. `evaluation.txt` contains precision, recall, F1, and the confusion matrix.

## Interpretation and limitation
Importance describes impurity reduction by features in this particular fitted tree; it does not show cause and effect. This historical dataset and model are for learning, not real-world decisions.
"""
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\nSaved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()



