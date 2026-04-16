import numpy as np


# -------------------------
# SINGLE PREDICTION
# -------------------------
def predict_pair(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    feats = pair_features(text_A, text_B, vocab)

    X = np.array([[feats[k] for k in feature_keys]])

    # same preprocessing as training
    X = np.clip(X, -10, 10)
    X = scaler.transform(X)

    pred = model.predict(X)[0]

    return pred


# -------------------------
# SYMMETRY CHECK (IMPORTANT)
# -------------------------
def symmetric_predict(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    pred1 = predict_pair(text_A, text_B, model, scaler, feature_keys, vocab, pair_features)
    pred2 = predict_pair(text_B, text_A, model, scaler, feature_keys, vocab, pair_features)

    # flip second prediction
    pred2 = 1 if pred2 == 2 else 2

    if pred1 == pred2:
        return pred1

    # fallback using KL divergence
    feats = pair_features(text_A, text_B, vocab)

    if feats["kl_AB"] > feats["kl_BA"]:
        return 2
    else:
        return 1


# -------------------------
# FINAL PREDICTION FUNCTION
# -------------------------
def final_predict(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    pred = symmetric_predict(
        text_A,
        text_B,
        model,
        scaler,
        feature_keys,
        vocab,
        pair_features
    )

    return pred