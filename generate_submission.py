import os
import pandas as pd

from src.features.feature_engineering import pair_features
from src.models.train import train_model
from src.inference.predict import final_predict
from src.data.loader import load_dataset


# -------------------------
# TRAIN MODEL FIRST
# -------------------------
print("🚀 Training model...")
df = load_dataset()

model, scaler, feature_keys, vocab, _ = train_model(df, pair_features)


# -------------------------
# LOAD TEST DATA
# -------------------------
test_path = "data/raw/test"

# filter only valid folders
test_ids = sorted([f for f in os.listdir(test_path) if f.startswith("article_")])


predictions = []

for article in test_ids:

    folder = os.path.join(test_path, article)

    file_A = os.path.join(folder, "file_1.txt")
    file_B = os.path.join(folder, "file_2.txt")

    # safety check
    if not os.path.exists(file_A) or not os.path.exists(file_B):
        print(f"⚠️ Skipping {article}, missing files")
        continue

    with open(file_A, encoding="utf-8") as f:
        text_A = f.read()

    with open(file_B, encoding="utf-8") as f:
        text_B = f.read()

    pred = final_predict(
        text_A,
        text_B,
        model,
        scaler,
        feature_keys,
        vocab,
        pair_features
    )

    # safer id extraction
    idx = int(article.replace("article_", ""))

    predictions.append([idx, pred])


# -------------------------
# SAVE CSV
# -------------------------
submission = pd.DataFrame(predictions, columns=["id", "label"])
submission = submission.sort_values("id")

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")