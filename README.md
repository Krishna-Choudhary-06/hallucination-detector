# 🧠 Hallucination Detection System (No Pretrained Models)

## 🚀 Problem
Given two summaries (Text A, Text B), detect which one is hallucinated.

Output:
- 1 → A is hallucinated
- 2 → B is hallucinated

---

## ⚙️ Approach

We use a **pairwise comparison framework** based on:

- KL Divergence (distribution mismatch)
- Entropy (randomness)
- Entity overlap
- Number inconsistency
- Structural differences

---

## 🧠 Key Idea

Instead of verifying facts externally, we detect hallucination by identifying **statistical and structural inconsistencies between two texts**.

---

## 🏗️ Pipeline

1. Feature Engineering (from scratch)
2. Pairwise comparison (A vs B)
3. Logistic Regression model
4. Symmetry training
5. Inference with consistency checks

---

## ❌ Constraints

- No pretrained models
- No embeddings
- Fully interpretable system

---

## 📊 Performance

- Validation Accuracy: ~0.81–0.85
- Robust across folds

---

## 💡 Why This Works

Hallucinated text often:
- has inconsistent numbers
- shows distribution shift
- has higher entropy
- breaks structural patterns

---

## 🏆 Highlights

- Fully from scratch
- Lightweight & explainable
- Competition-compliant