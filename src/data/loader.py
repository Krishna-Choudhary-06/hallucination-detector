import os
import pandas as pd


def load_dataset():
    meta = pd.read_csv("data/raw/train.csv")
    base_path = "data/raw/train"

    data = []

    for _, row in meta.iterrows():
        article_id = f"article_{int(row['id']):04d}"
        real_id = int(row["real_text_id"])

        folder = os.path.join(base_path, article_id)

        with open(os.path.join(folder, "file_1.txt"), encoding="utf-8") as f:
            text_A = f.read()

        with open(os.path.join(folder, "file_2.txt"), encoding="utf-8") as f:
            text_B = f.read()

        # label conversion
        label = 2 if real_id == 1 else 1

        data.append({
            "text_A": text_A,
            "text_B": text_B,
            "label": label
        })

    df = pd.DataFrame(data)
    return df