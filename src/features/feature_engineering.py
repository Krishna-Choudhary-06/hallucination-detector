import re
import numpy as np
from collections import Counter


# -------------------------
# BASIC FEATURES
# -------------------------
def basic_features(text):
    words = text.split()
    sentences = re.split(r'[.!?]', text)

    word_count = len(words)
    sent_count = len(sentences)

    avg_sent_len = word_count / (sent_count + 1)
    unique_ratio = len(set(words)) / (word_count + 1)

    return {
        "word_count": word_count,
        "avg_sent_len": avg_sent_len,
        "unique_ratio": unique_ratio,
    }


# -------------------------
# ENTITY OVERLAP
# -------------------------
def entity_overlap(a, b):
    entA = set([w for w in a.split() if w.istitle()])
    entB = set([w for w in b.split() if w.istitle()])
    return len(entA & entB) / (len(entA | entB) + 1e-5)


# -------------------------
# ENTROPY
# -------------------------
def entropy(text):
    words = text.split()
    if len(words) == 0:
        return 0.0

    counts = Counter(words)
    probs = np.array(list(counts.values())) / len(words)

    return -np.sum(probs * np.log(probs + 1e-9))


# -------------------------
# KL DIVERGENCE
# -------------------------
def kl_divergence(text_A, text_B):
    words_A = text_A.split()
    words_B = text_B.split()

    if len(words_A) == 0 or len(words_B) == 0:
        return 0.0

    freq_A = Counter(words_A)
    freq_B = Counter(words_B)

    total_A = len(words_A)
    total_B = len(words_B)

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
# SENTENCE COUNT DIFFERENCE
# -------------------------
def sentence_count_diff(A, B):
    return abs(len(A.split('.')) - len(B.split('.')))


# -------------------------
# FINAL PAIR FEATURES
# -------------------------
def pair_features(text_A, text_B):

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
    # HIGH-SIGNAL FEATURES
    # -------------------------
    features["entity_overlap"] = entity_overlap(text_A, text_B)

    features["entropy_diff"] = np.tanh(
        entropy(text_A) - entropy(text_B)
    )

    features["kl_AB"] = kl_divergence(text_A, text_B)
    features["kl_BA"] = kl_divergence(text_B, text_A)

    # -------------------------
    # FACTUAL + STRUCTURAL
    # -------------------------
    features["number_diff"] = number_diff(text_A, text_B)
    features["sentence_count_diff"] = sentence_count_diff(text_A, text_B)

    features["length_ratio"] = (len(text_A.split()) + 1) / (len(text_B.split()) + 1)

    # -------------------------
    # SINGLE STRONG INTERACTION
    # -------------------------
    features["kl_number_interaction"] = (
        features["kl_AB"] * features["number_diff"]
    )

    return features