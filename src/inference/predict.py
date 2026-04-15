import numpy as np
import re


# -------------------------
# SINGLE PREDICTION
# -------------------------
def predict_pair(text_A, text_B, model, scaler, feature_keys, pair_features):

    feats = pair_features(text_A, text_B)
    X = np.array([[feats[k] for k in feature_keys]])

    X = np.clip(X, -10, 10)
    X = scaler.transform(X)

    pred = model.predict(X)[0]

    return pred


# -------------------------
# RULE BOOST
# -------------------------
def rule_boost(text_A, text_B, base_pred):

    nums_A = len(re.findall(r'\d+', text_A))
    nums_B = len(re.findall(r'\d+', text_B))

    if abs(nums_A - nums_B) >= 2:
        return 1 if nums_A < nums_B else 2

    return base_pred


# -------------------------
# SYMMETRY CHECK
# -------------------------
def symmetric_predict(text_A, text_B, model, scaler, feature_keys, pair_features):

    pred1 = predict_pair(text_A, text_B, model, scaler, feature_keys, pair_features)
    pred2 = predict_pair(text_B, text_A, model, scaler, feature_keys, pair_features)

    pred2 = 1 if pred2 == 2 else 2

    if pred1 == pred2:
        return pred1

    feats = pair_features(text_A, text_B)

    if feats["kl_AB"] > feats["kl_BA"]:
        return 2
    else:
        return 1


# -------------------------
# FINAL PREDICTION
# -------------------------
def final_predict(text_A, text_B, model, scaler, feature_keys, pair_features):

    pred = symmetric_predict(text_A, text_B, model, scaler, feature_keys, pair_features)
    pred = rule_boost(text_A, text_B, pred)

    return pred