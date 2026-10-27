# Titanic Survival with a Decision Tree

A beginner-friendly classification project using Pandas and scikit-learn to predict Titanic passenger survival from **age, sex, passenger class, and fare**.

## What it does

- Loads Kaggle's Titanic competition `train.csv` when available; otherwise downloads a public copy of the Titanic passenger data.
- Uses Pandas to inspect dimensions, survival counts, and missing values.
- Imputes missing numeric values with training-set medians and missing categorical values with the most frequent category.
- One-hot encodes `sex`, then trains a scikit-learn decision tree.
- Evaluates on a stratified 20% holdout set and prints accuracy, a confusion matrix, and precision/recall/F1.
- Saves a feature-importance CSV and chart under `results/`.

## Requirements

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python titanic_tree.py
```

By default, the script downloads the public CSV to `data/titanic.csv`. To use Kaggle's competition data instead, download `train.csv` from [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) and place it at `data/train.csv`; Kaggle's file takes precedence.

Expected outputs:

- `results/evaluation.txt` — test accuracy, confusion matrix, and classification report
- `results/report.md` — generated step-by-step report with the measured accuracy and top feature
- `results/evaluation_summary.png` — evaluation output snapshot for the report
- `results/feature_importance.csv` — decision-tree importances for the transformed features
- `results/feature_importance.png` — feature-importance chart (include this and the evaluation snapshot as output screenshots)

## Method

1. Select `age`, `sex`, `pclass`, and `fare` as predictors and `survived` as the label.
2. Split the examples into 80% training and 20% testing data, with the same survival-class proportions in both sets.
3. Put imputation, encoding, and the model in one pipeline. This ensures preprocessing is fit on the training portion only.
4. Train a tree limited to depth 4 with a minimum of 5 examples in a leaf to keep it interpretable and reduce overfitting.
5. Report holdout metrics and the model's impurity-based feature importances.

## Interpreting feature importance

Importance values describe how much each input reduced impurity across this particular fitted tree. They are model-specific and are not causal explanations. Sex may appear as separate one-hot encoded categories (for example, `sex_female` and `sex_male`). Small changes to the split or tree settings can change the values.

## Limitations

This is a small educational dataset and a simple model. Accuracy is the fraction of test passengers classified correctly; it does not mean the model is equally good for every passenger subgroup. The model is not suitable for real-world survival decisions. Results can vary if the input data or split changes.

## Data source

- [Kaggle Titanic competition data](https://www.kaggle.com/competitions/titanic/data)
- Fallback public CSV: [seaborn-data Titanic CSV](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)



## Project report

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the step-by-step write-up and screenshot instructions. Running the script creates the measured report and output images under esults/.
