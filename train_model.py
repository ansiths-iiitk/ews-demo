"""
Step 1 of the demo: TRAINING.
Builds a small demonstration dataset, trains a model, and saves it to a file.

Run once:   python train_model.py
Creates:    model.joblib         (the trained model)
            training_data.csv    (the data it learned from; used by the drift check)

The data is synthetic (made up for teaching). It is NOT real student data.
Marks are out of 100, attendance is a percentage.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression


def train(n=600, seed=42):
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(80, 12, n), 35, 100).round(0)
    # students who attend more tend to score more
    marks = np.clip(0.6 * attendance + rng.normal(15, 14, n), 5, 100).round(0)

    # "needs review" becomes more likely when attendance and marks are low
    score = 0.09 * (70 - attendance) + 0.08 * (50 - marks) + rng.normal(0, 1, n)
    needs_review = (score > 0).astype(int)

    data = pd.DataFrame({"attendance": attendance, "marks": marks, "needs_review": needs_review})
    data.to_csv("training_data.csv", index=False)

    model = LogisticRegression()
    model.fit(data[["attendance", "marks"]], data["needs_review"])
    joblib.dump(model, "model.joblib")

    print(f"Trained on {n} students; {needs_review.mean():.0%} labelled 'needs review'.")
    print("Saved model.joblib and training_data.csv")


if __name__ == "__main__":
    train()
