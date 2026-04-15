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

model, scaler, feature_keys, _ = train_model(df, pair_features)


# -------------------------
# LOAD TEST DATA
# -------------------------
test_path = "data/raw/test"
test_ids = sorted(os.listdir(test_path))


predictions = []

for article in test_ids:

    folder = os.path.join(test_path, article)

    with open(os.path.join(folder, "file_1.txt"), encoding="utf-8") as f:
        text_A = f.read()

    with open(os.path.join(folder, "file_2.txt"), encoding="utf-8") as f:
        text_B = f.read()

    pred = final_predict(text_A, text_B, model, scaler, feature_keys, pair_features)

    idx = int(article.split("_")[1])

    predictions.append([idx, pred])


# -------------------------
# SAVE CSV
# -------------------------
submission = pd.DataFrame(predictions, columns=["id", "label"])
submission = submission.sort_values("id")

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")