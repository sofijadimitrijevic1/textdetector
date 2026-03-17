"""
Sanity-check classifier: predict AI vs human using only a fixed list of
buzzwords identified by the TF-IDF logistic regression.

If this scores nearly as well as BERT, the task is trivially solvable by
word counting — the dataset has a vocabulary bias problem, not a real signal.

Usage:
    python buzzword_classifier.py --data data/raw/arxiv_pairs_1000.csv
"""

import argparse
import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Buzzword lists — taken directly from TF-IDF logistic regression top features
# These are the words with the highest absolute coefficients.
# ---------------------------------------------------------------------------

# Words that strongly predict AI text (positive coefficients)
AI_WORDS = [
    "novel", "robust", "critical", "diverse", "often",
    "significantly", "significant", "crucial", "superior",
    "comprehensive", "consistently", "findings", "substantial",
    "particularly", "leveraging", "critically", "innovative",
    "promising", "notably", "demonstrating", "state-of-the-art",
    "furthermore", "additionally", "importantly", "notably",
    "highlighting", "showcasing", "outperforms", "advancing",
]

# Words that strongly predict HUMAN text (negative coefficients)
# Add any you observed from the "Top phrases predicting HUMAN" output
HUMAN_WORDS = [
    # Fill these in from your actual output — the negative-coef words
    # e.g. field-specific abbreviations, passive constructions, etc.
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def count_buzzwords(text: str, wordlist: list[str]) -> int:
    """Count how many buzzword occurrences appear in text (case-insensitive)."""
    text_lower = text.lower()
    total = 0
    for word in wordlist:
        # whole-word match to avoid e.g. "significant" matching "insignificant"
        total += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
    return total


def predict(text: str) -> tuple[int, dict]:
    """
    Returns predicted label (0=human, 1=AI) and a dict of scores.
    Decision rule: more AI buzzwords than human buzzwords → predict AI.
    Ties go to human (conservative).
    """
    ai_score    = count_buzzwords(text, AI_WORDS)
    human_score = count_buzzwords(text, HUMAN_WORDS)
    label = 1 if ai_score > human_score else 0
    return label, {"ai_score": ai_score, "human_score": human_score}


# ---------------------------------------------------------------------------
# Data loading — mirrors the notebook's pair-based split exactly
# ---------------------------------------------------------------------------

def load_and_split(csv_path: str):
    pairs = pd.read_csv(csv_path).dropna(subset=["real_abstract", "ai_abstract"])

    train_pairs, temp   = train_test_split(pairs, test_size=0.2, random_state=42)
    val_pairs,   test_p = train_test_split(temp,  test_size=0.5, random_state=42)

    def flatten(df):
        human = pd.DataFrame({"text": df["real_abstract"].values, "label": 0})
        ai    = pd.DataFrame({"text": df["ai_abstract"].values,   "label": 1})
        return pd.concat([human, ai], ignore_index=True)

    return flatten(train_pairs), flatten(val_pairs), flatten(test_p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str):
    print(f"Loading: {csv_path}")
    _, _, test_df = load_and_split(csv_path)
    print(f"Test set: {len(test_df)} rows ({(test_df.label==0).sum()} human, {(test_df.label==1).sum()} AI)\n")

    preds, ai_scores, human_scores = [], [], []
    for text in test_df["text"]:
        label, scores = predict(text)
        preds.append(label)
        ai_scores.append(scores["ai_score"])
        human_scores.append(scores["human_score"])

    acc = accuracy_score(test_df["label"], preds)
    print(f"Buzzword classifier accuracy: {acc:.4f}\n")
    print(classification_report(test_df["label"], preds, target_names=["human", "ai"], digits=4))

    # Show average buzzword counts per class so you can see the gap
    test_df = test_df.copy()
    test_df["ai_score"]    = ai_scores
    test_df["human_score"] = human_scores

    print("── Average AI-buzzword count per true class ──")
    print(test_df.groupby("label")[["ai_score", "human_score"]].mean().rename(index={0: "human", 1: "ai"}))

    # Show the 5 texts with the most AI buzzwords that are actually human
    print("\n── Human texts with the most AI buzzwords (false-positive candidates) ──")
    human_texts = test_df[test_df.label == 0].nlargest(5, "ai_score")
    for _, row in human_texts.iterrows():
        print(f"\n  ai_score={row['ai_score']}  text: {row['text'][:200]}...")

    # Show which buzzwords are actually firing most often
    print("\n── Buzzword hit frequency in test set ──")
    all_text = " ".join(test_df["text"].str.lower())
    hits = {w: len(re.findall(r'\b' + re.escape(w) + r'\b', all_text)) for w in AI_WORDS}
    for word, count in sorted(hits.items(), key=lambda x: -x[1]):
        print(f"  {word:<30} {count} hits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/arxiv_pairs_1000.csv")
    args = parser.parse_args()
    main(args.data)
