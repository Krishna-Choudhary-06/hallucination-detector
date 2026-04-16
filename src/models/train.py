import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.vocab import build_vocab


def train_model(df, pair_features):

    X = []
    y = []

    feature_keys = None

    # -------------------------
    # BUILD VOCAB
    # -------------------------
    all_texts = []
    for _, row in df.iterrows():
        all_texts.append(row["text_A"])
        all_texts.append(row["text_B"])

    vocab = build_vocab(all_texts, max_features=3000)

    # -------------------------
    # BUILD DATA (SYMMETRY)
    # -------------------------
    for _, row in df.iterrows():

        feats = pair_features(row["text_A"], row["text_B"], vocab)

        if feature_keys is None:
            feature_keys = list(feats.keys())

        X.append([feats[k] for k in feature_keys])
        y.append(row["label"])

        # reverse
        rev_feats = pair_features(row["text_B"], row["text_A"], vocab)
        X.append([rev_feats[k] for k in feature_keys])

        rev_label = 1 if row["label"] == 2 else 2
        y.append(rev_label)

    X = np.array(X)
    y = np.array(y)

    # -------------------------
    # CLIP + SCALE
    # -------------------------
    X = np.clip(X, -10, 10)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------
    # FEATURE WEIGHTING 🔥
    # -------------------------
    weights = []
    for key in feature_keys:
        if "kl" in key:
            weights.append(2.0)
        elif "entropy" in key:
            weights.append(1.8)
        elif "bow" in key:
            weights.append(2.2)
        elif "length" in key:
            weights.append(1.5)
        else:
            weights.append(1.0)

    weights = np.array(weights)
    X = X * weights

    # -------------------------
    # K-FOLD
    # -------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_model = None
        best_score = 0

        for C in [1.5, 2.0, 3.0]:

            model = LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced"
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)

            if acc > best_score:
                best_score = acc
                best_model = model

        print(f"Fold {fold} → Accuracy: {best_score:.3f}")
        print("------------------------------------")

        scores.append(best_score)

    final_score = np.mean(scores)

    print("🔥 Final Average Accuracy:", final_score)

    return best_model, scaler, feature_keys, vocab, final_score