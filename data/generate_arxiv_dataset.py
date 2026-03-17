import argparse
import os
import tempfile

import arxiv
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

CATEGORIES = ["cs.CV", "cs.CL", "physics.optics", "q-bio.GN", "econ.EM"]

MIN_DATE = "2024-06-01"
MAX_DATE = (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y%m%d")

MAX_BODY_WORDS = 8000

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "processed")
PAIRS_CSV = os.path.join(RAW_DIR, "arxiv_pairs_1000.csv")
OUT_CSV = os.path.join(PROC_DIR, "dataset_1000.csv")


def fetch_papers(category: str, n: int) -> list:
    print(f"Fetching {category}...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category} AND submittedDate:[{MIN_DATE.replace('-', '')} TO {MAX_DATE}]",
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = list(client.results(search))
    print(f"  Got {len(papers)} papers.")
    return papers


def download_body(paper) -> str | None:
    print("  Downloading PDF...", end=" ", flush=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            response = requests.get(paper.pdf_url, timeout=30)
            f.write(response.content)
            tmp_path = f.name

        full_text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"

        os.unlink(tmp_path)
    except Exception as e:
        print(f"failed: {e}")
        return None

    lower = full_text.lower()
    for marker in ["1 introduction", "1. introduction", "introduction\n"]:
        idx = lower.find(marker)
        if idx != -1:
            body = full_text[idx:]
            break
    else:
        body = " ".join(full_text.split()[300:])

    for marker in ["\nreferences\n", "\nreferences \n", "\nbibliography\n"]:
        ref_idx = body.lower().find(marker)
        if ref_idx != -1:
            body = body[:ref_idx]
            break

    word_count = len(body.split())

    if word_count < 200:
        print(f"too short ({word_count} words), skipping.")
        return None

    if word_count > MAX_BODY_WORDS:
        print(f"too long ({word_count} words), skipping.")
        return None

    print(f"ok ({word_count} words).")
    return body


def call_gemini(client, system: str, user: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system}\n\n{user}",
    )
    return response.text.strip()


STAGE1_SYSTEM = (
    "You are an academic writing analyst. "
    "Analyze writing style only — never reproduce or paraphrase specific results, numbers, or findings."
)

STAGE1_USER = """\
Analyze the writing style of this academic abstract. Describe:
- Overall structure and flow
- Tone and level of formality
- How hedging language is used (e.g. "we propose", "suggest", "to the best of our knowledge")
- Sentence length and complexity
- How results are framed (without stating what the results actually are)

Do NOT mention any specific results, numbers, methods, or datasets from the abstract.

Abstract:
{abstract}"""

STAGE2_SYSTEM = (
    "You are an academic abstract writer. "
    "Write only the abstract text — no title, no labels, no commentary."
)

STAGE2_USER = """\
Write an academic abstract of approximately {n_words} words for a paper with the following body content.

Style guide (match this closely):
{style_guide}

Paper body:
{body}"""


def load_done() -> set:
    if not os.path.exists(PAIRS_CSV):
        return set()
    return set(pd.read_csv(PAIRS_CSV)["arxiv_id"].tolist())


def run(papers_per_category: int, max_pairs: int):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    done = load_done()

    print(f"Already done: {len(done)} | target: {max_pairs}")

    os.makedirs(RAW_DIR, exist_ok=True)
    write_header = not os.path.exists(PAIRS_CSV)
    generated = 0
    skipped = 0
    failed = 0

    for category in CATEGORIES:
        papers = fetch_papers(category, papers_per_category)

        for paper in papers:
            if generated >= max_pairs:
                break

            arxiv_id = paper.entry_id.split("/")[-1]
            print(f"\n[{generated + 1}/{max_pairs}] {paper.title[:70]}")

            if arxiv_id in done:
                skipped += 1
                continue

            real_abstract = paper.summary.replace("\n", " ").strip()
            target_wc = len(real_abstract.split())

            body = download_body(paper)
            if body is None:
                failed += 1
                continue

            try:
                print("  Stage 1...", end=" ", flush=True)
                style_guide = call_gemini(
                    client, STAGE1_SYSTEM, STAGE1_USER.format(abstract=real_abstract)
                )
                print("done.")

                print("  Stage 2...", end=" ", flush=True)
                ai_abstract = call_gemini(
                    client,
                    STAGE2_SYSTEM,
                    STAGE2_USER.format(
                        n_words=target_wc, style_guide=style_guide, body=body
                    ),
                )
                words = ai_abstract.split()
                if len(words) > target_wc + 15:
                    ai_abstract = " ".join(words[:target_wc])
                print(f"done ({len(ai_abstract.split())} words).")
            except Exception as e:
                print(f"  LLM error: {e}")
                failed += 1
                continue

            record = pd.DataFrame(
                [
                    {
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "category": category,
                        "published": str(paper.published.date()),
                        "real_abstract": real_abstract,
                        "ai_abstract": ai_abstract,
                        "real_wc": target_wc,
                        "ai_wc": len(ai_abstract.split()),
                    }
                ]
            )

            record.to_csv(PAIRS_CSV, mode="a", header=write_header, index=False)
            write_header = False
            generated += 1
            done.add(arxiv_id)

        if generated >= max_pairs:
            break

    print(f"\nDone. generated={generated} skipped={skipped} failed={failed}")


def merge():
    pairs = pd.read_csv(PAIRS_CSV)
    print(f"\nMerging {len(pairs)} pairs into dataset...")

    human_rows = pairs[["real_abstract", "category", "published"]].copy()
    human_rows = human_rows.rename(columns={"real_abstract": "text"})
    human_rows["label"] = 0

    ai_rows = pairs[["ai_abstract", "category", "published"]].copy()
    ai_rows = ai_rows.rename(columns={"ai_abstract": "text"})
    ai_rows["label"] = 1

    combined = pd.concat([human_rows, ai_rows], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(PROC_DIR, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(combined)} rows to {OUT_CSV}")
    print(f"  Human : {(combined['label'] == 0).sum()}")
    print(f"  AI    : {(combined['label'] == 1).sum()}")

    combined["wc"] = combined["text"].str.split().str.len()
    print("\nWord count by label:")
    print(combined.groupby("label")["wc"].describe()[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers-per-category",
        type=int,
        default=250,
        help="Papers to fetch per arXiv category (fetches more than needed to account for skips)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        help="Stop after generating this many pairs (default 1000)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip generation and only rebuild dataset.csv from existing pairs",
    )
    args = parser.parse_args()

    if not args.merge_only:
        run(args.papers_per_category, args.max_pairs)

    if os.path.exists(PAIRS_CSV):
        merge()
    else:
        print("No pairs CSV found. Run without --merge-only first.")
