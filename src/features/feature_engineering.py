import re
import numpy as np
from collections import Counter

from src.features.tokenizer import tokenize
from src.features.vocab import text_to_vector


# -------------------------
# BASIC FEATURES
# -------------------------
def basic_features(text):
    tokens = tokenize(text)

    word_count = len(tokens)

    sentences = re.split(r'[.!?]', text)
    sent_count = len(sentences)

    avg_sent_len = word_count / (sent_count + 1)
    unique_ratio = len(set(tokens)) / (word_count + 1)

    return {
        "word_count": word_count,
        "avg_sent_len": avg_sent_len,
        "unique_ratio": unique_ratio,
    }


# -------------------------
# ENTROPY
# -------------------------
def entropy(text):
    tokens = tokenize(text)

    if len(tokens) == 0:
        return 0.0

    counts = Counter(tokens)
    probs = np.array(list(counts.values())) / len(tokens)

    return -np.sum(probs * np.log(probs + 1e-9))


# -------------------------
# KL DIVERGENCE
# -------------------------
def kl_divergence(text_A, text_B):
    tokens_A = tokenize(text_A)
    tokens_B = tokenize(text_B)

    if len(tokens_A) == 0 or len(tokens_B) == 0:
        return 0.0

    freq_A = Counter(tokens_A)
    freq_B = Counter(tokens_B)

    total_A = len(tokens_A)
    total_B = len(tokens_B)

    vocab = set(freq_A.keys()).union(set(freq_B.keys()))

    epsilon = 1e-9
    kl = 0.0

    for word in vocab:
        p = (freq_A.get(word, 0) + epsilon) / total_A
        q = (freq_B.get(word, 0) + epsilon) / total_B

        kl += p * np.log(p / q)

    return kl


# -------------------------
# NUMBER DIFFERENCE
# -------------------------
def number_diff(A, B):
    nums_A = re.findall(r'\d+', A)
    nums_B = re.findall(r'\d+', B)
    return abs(len(nums_A) - len(nums_B))


# -------------------------
# FINAL PAIR FEATURES
# -------------------------
def pair_features(text_A, text_B, vocab):

    fA = basic_features(text_A)
    fB = basic_features(text_B)

    features = {}

    # -------------------------
    # CORE FEATURES
    # -------------------------
    features["word_count_diff"] = fA["word_count"] - fB["word_count"]
    features["unique_ratio_diff"] = fA["unique_ratio"] - fB["unique_ratio"]
    features["avg_sent_len_diff"] = fA["avg_sent_len"] - fB["avg_sent_len"]

    # -------------------------
    # STRONG STATISTICAL SIGNALS
    # -------------------------
    features["entropy_diff"] = np.tanh(
        entropy(text_A) - entropy(text_B)
    )

    features["kl_AB"] = kl_divergence(text_A, text_B)
    features["kl_BA"] = kl_divergence(text_B, text_A)

    # -------------------------
    # FACTUAL SIGNAL
    # -------------------------
    features["number_diff"] = number_diff(text_A, text_B)

    # -------------------------
    # STRUCTURE SIGNAL
    # -------------------------
    features["length_ratio"] = (len(tokenize(text_A)) + 1) / (len(tokenize(text_B)) + 1)

    features["length_diff"] = len(text_A) - len(text_B)   # 🔥 NEW STRONG

    # -------------------------
    # LEXICAL SIGNAL (VERY IMPORTANT)
    # -------------------------
    vec_A = text_to_vector(text_A, vocab)
    vec_B = text_to_vector(text_B, vocab)

    features["bow_diff"] = sum(abs(a - b) for a, b in zip(vec_A, vec_B))

    # -------------------------
    # FINAL HIGH-SIGNAL INTERACTION
    # -------------------------
    features["kl_number_interaction"] = features["kl_AB"] * features["number_diff"]

    return features