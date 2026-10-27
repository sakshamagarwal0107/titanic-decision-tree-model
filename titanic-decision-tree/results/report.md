# Titanic Survival Decision Tree Report

## Objective
Use age, sex, passenger class, and fare to predict Titanic passenger survival.

## Steps
1. Loaded 891 rows from public seaborn Titanic CSV with Pandas and inspected dimensions, null counts, and survival counts.
2. Imputed missing age/fare values with training-set medians and missing sex with the training-set mode.
3. One-hot encoded sex and retained age, passenger class, and fare as numeric predictors.
4. Split into 712 training and 179 test passengers (80/20, stratified, random_state=42).
5. Fit a depth-limited scikit-learn decision tree and measured its performance on the held-out test set.

## Results
- **Accuracy:** 76.5% (0.765) on 179 held-out passengers.
- **Top feature in this fitted tree:** `sex_female` (importance 0.607).
- **Confusion matrix:** see `evaluation.txt`; rows are actual [did not survive, survived] and columns predicted [did not survive, survived].

See `evaluation_summary.png` for a visual evaluation snapshot and `feature_importance.png` for the importance chart. `evaluation.txt` contains precision, recall, F1, and the confusion matrix.

## Interpretation and limitation
Importance describes impurity reduction by features in this particular fitted tree; it does not show cause and effect. This historical dataset and model are for learning, not real-world decisions.
