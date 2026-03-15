"""
View arxiv pairs in a readable format.

Usage:
  python data/view_pairs.py           # show all pairs
  python data/view_pairs.py --n 5     # show first 5
  python data/view_pairs.py --i 2     # show specific pair by index
"""

import argparse
import os

import pandas as pd

PAIRS_CSV = os.path.join(os.path.dirname(__file__), "raw", "arxiv_pairs_1000.csv")

SEP = "=" * 80
SEP2 = "-" * 80


def show_pair(row, idx):
    print(SEP)
    print(
        f"  #{idx}  |  {row['category']}  |  {row['published']}  |  {row['arxiv_id']}"
    )
    print(SEP)
    print(f"TITLE: {row['title']}")
    print()
    print(f"HUMAN ABSTRACT  ({row['real_wc']} words)")
    print(SEP2)
    print(row["real_abstract"])
    print()
    print(f"AI ABSTRACT  ({row['ai_wc']} words)")
    print(SEP2)
    print(row["ai_abstract"])
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="Show first N pairs (0 = all)")
    parser.add_argument(
        "--i", type=int, default=-1, help="Show a specific pair by index"
    )
    args = parser.parse_args()

    df = pd.read_csv(PAIRS_CSV)
    print(f"Total pairs: {len(df)}\n")

    if args.i >= 0:
        show_pair(df.iloc[args.i], args.i)
    else:
        subset = df.head(args.n) if args.n > 0 else df
        for idx, row in subset.iterrows():
            show_pair(row, idx)


if __name__ == "__main__":
    main()
