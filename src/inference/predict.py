import numpy as np


# -------------------------
# SINGLE PREDICTION (UPDATED)
# -------------------------
def predict_pair(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    # extract features
    feats = pair_features(text_A, text_B, vocab)

    X = np.array([[feats[k] for k in feature_keys]])

    # same preprocessing as training
    X = np.clip(X, -10, 10)
    X = scaler.transform(X)

    # -------------------------
    # 🔥 NEW: PROBABILITY + CONFIDENCE
    # -------------------------
    proba = model.predict_proba(X)[0]   # <-- UPDATED

    confidence = max(proba)             # <-- UPDATED

    pred = model.predict(X)[0]

    # -------------------------
    # 🔥 NEW: LOW CONFIDENCE FALLBACK
    # -------------------------
    if confidence < 0.6:                # <-- UPDATED
        if feats["kl_AB"] > feats["kl_BA"]:
            return 2
        else:
            return 1

    return pred


# -------------------------
# SYMMETRY CHECK (UPDATED)
# -------------------------
def symmetric_predict(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    pred1 = predict_pair(text_A, text_B, model, scaler, feature_keys, vocab, pair_features)

    pred2 = predict_pair(text_B, text_A, model, scaler, feature_keys, vocab, pair_features)

    # flip second prediction
    pred2 = 1 if pred2 == 2 else 2

    # -------------------------
    # 🔥 UPDATED: STRICT SYMMETRY RULE
    # -------------------------
    if pred1 == pred2:
        return pred1

    # fallback using KL divergence
    feats = pair_features(text_A, text_B, vocab)

    if feats["kl_AB"] > feats["kl_BA"]:
        return 2
    else:
        return 1


# -------------------------
# FINAL PREDICTION
# -------------------------
def final_predict(text_A, text_B, model, scaler, feature_keys, vocab, pair_features):

    feats = pair_features(text_A, text_B, vocab)

    # -------------------------
    # 🔥 OPTIONAL: AMBIGUITY DETECTION
    # -------------------------
    if abs(feats["kl_AB"] - feats["kl_BA"]) < 0.05:
        # you can return 1 or 2 OR mark uncertain
        # for competition → must return 1/2
        return 1  # <-- UPDATED (safe fallback)

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