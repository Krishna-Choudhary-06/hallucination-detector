import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_model(df, pair_features):

    X = []
    y = []

    feature_keys = None

    # -------------------------
    # BUILD DATA (SYMMETRY)
    # -------------------------
    for _, row in df.iterrows():

        feats = pair_features(row["text_A"], row["text_B"])

        if feature_keys is None:
            feature_keys = list(feats.keys())

        X.append([feats[k] for k in feature_keys])
        y.append(row["label"])

        # reverse
        rev_feats = pair_features(row["text_B"], row["text_A"])
        X.append([rev_feats[k] for k in feature_keys])

        rev_label = 1 if row["label"] == 2 else 2
        y.append(rev_label)

    X = np.array(X)
    y = np.array(y)

    # stability
    X = np.clip(X, -10, 10)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 🔥 FINAL MODEL
        model = LogisticRegression(C=2.0, max_iter=1000)

        model.fit(X_train, y_train)

        # 🔥 NORMAL PREDICTION (NO CONFIDENCE TRICK)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)

        print(f"Fold {fold} → Accuracy: {acc:.3f}")
        print("------------------------------------")

        scores.append(acc)

    final_score = np.mean(scores)

    print("🔥 Final Average Accuracy:", final_score)

    return model, scaler, feature_keys, final_score