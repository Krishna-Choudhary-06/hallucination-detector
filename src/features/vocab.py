from collections import Counter
from pyexpat import features
from src.features.tokenizer import tokenize



def build_vocab(texts, max_features=3000):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {word: i for i, (word, _) in enumerate(counter.most_common(max_features))}

    return vocab


def text_to_vector(text, vocab):
    tokens = tokenize(text)
    vector = [0] * len(vocab)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1

    return vector

# -------------------------
# BAG OF WORD DIFFERENCE (NEW)
# -------------------------
def pair_features(text_A, text_B, vocab):
    vec_A = text_to_vector(text_A, vocab)
    vec_B = text_to_vector(text_B, vocab)

    bow_diff = sum(abs(a - b) for a, b in zip(vec_A, vec_B))

    features["bow_diff"] = bow_diff
