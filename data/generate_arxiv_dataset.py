"""
Generate a paired human/AI abstract dataset from arXiv papers.

Pipeline per paper:
  1. Fetch recent paper from arXiv (title, real abstract, PDF)
  2. Extract body text from PDF
  3. Stage 1 — ask LLM to analyze the abstract's writing style (no content leaked)
  4. Stage 2 — ask LLM to write a new abstract using body + style guide
  5. Hard-truncate generated abstract to match real abstract word count

Output:
  data/raw/arxiv_pairs.csv      - paired (real_abstract, ai_abstract) with metadata
  data/processed/dataset.csv    - flat labeled dataset (text, label) ready for training

Usage:
  pip install arxiv pdfplumber anthropic python-dotenv
  python generate_arxiv_dataset.py --provider groq --papers-per-category 40
  python generate_arxiv_dataset.py --merge-only
"""

import argparse
import os

import time
import tempfile
from collections import deque

import arxiv
import pdfplumber
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Multiple domains to avoid field-bias
CATEGORIES = ["cs.CL", "cs.CV", "physics.optics", "q-bio.GN", "econ.EM"]

# Only fetch papers submitted after this date to avoid LLM regurgitation
MIN_DATE = "2024-06-01"

# How much body text to send (words) — enough context without blowing token limit
MAX_BODY_WORDS = 8000  # skip papers whose body exceeds this — don't truncate

# Gemini Pro limits
GEMINI_RPM = 2000

PAIRS_CSV = os.path.join(os.path.dirname(__file__), "raw",       "arxiv_pairs.csv")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "processed", "dataset.csv")

# Groq limits (llama-3.3-70b-versatile)
GROQ_RPM   = 30      # requests per minute
GROQ_TPM   = 12000   # tokens per minute (binding constraint)
GROQ_RPD   = 1000    # requests per day
GROQ_TPD   = 1_000_000  # tokens per day
# At ~3000 tokens per call pair and 12000 TPM, allow ~4 pairs/min → 15s between calls
GROQ_INTER_STAGE_DELAY = 15  # seconds to wait between stage 1 and stage 2


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Tracks API calls in a rolling 60-second window and sleeps with a
    countdown when approaching the requests-per-minute limit.
    """
    def __init__(self, rpm: int):
        self.rpm       = rpm
        self.min_gap   = 60.0 / rpm   # minimum seconds between calls
        self.calls     = deque()       # timestamps of recent calls
        self.total     = 0
        self.day_start = time.time()

    def wait(self):
        now = time.time()

        # Drop calls older than 60 seconds
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()

        # If at the per-minute limit, wait until the oldest call expires
        if len(self.calls) >= self.rpm:
            wait_until = self.calls[0] + 60
            remaining  = wait_until - time.time()
            if remaining > 0:
                print("    [rate limit] RPM cap reached — waiting ", end="", flush=True)
                while remaining > 0:
                    print(f"{int(remaining)}s ", end="", flush=True)
                    time.sleep(min(1, remaining))
                    remaining = wait_until - time.time()
                print()

        # Enforce minimum gap between consecutive calls
        if self.calls:
            gap = time.time() - self.calls[-1]
            if gap < self.min_gap:
                time.sleep(self.min_gap - gap)

        self.calls.append(time.time())
        self.total += 1

    def stats(self) -> str:
        elapsed_min = (time.time() - self.day_start) / 60
        return f"calls today: {self.total} | elapsed: {elapsed_min:.1f}min | remaining RPD: {GROQ_RPD - self.total}"


# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------

def fetch_papers(category: str, n: int) -> list:
    print(f"  Fetching up to {n} papers from arXiv [{category}]...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category} AND submittedDate:[{MIN_DATE.replace('-', '')} TO 99991231]",
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = list(client.results(search))
    print(f"  Got {len(papers)} papers.")
    return papers


def download_body(paper) -> str | None:
    """Download PDF and extract body text (introduction onwards, references stripped)."""
    print("    Downloading PDF...", end=" ", flush=True)
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
        print(f"FAILED ({e})")
        return None

    # Find introduction — skip title/abstract/metadata at the top
    lower = full_text.lower()
    for marker in ["1 introduction", "1. introduction", "introduction\n"]:
        idx = lower.find(marker)
        if idx != -1:
            body = full_text[idx:]
            break
    else:
        body = " ".join(full_text.split()[300:])

    # Strip references section
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
        print(f"too long ({word_count} words > {MAX_BODY_WORDS} limit), skipping.")
        return None

    print(f"OK ({word_count} words).")
    return body


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

def call_anthropic(client, system: str, user: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


def call_openai(client, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=600,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def call_gemini(client, system: str, user: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system}\n\n{user}",
    )
    return response.text.strip()


def call_groq(client, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=600,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def build_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "openai":
        import openai
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "groq":
        from groq import Groq
        return Groq(api_key=os.environ["GROQ_API_KEY"])
    elif provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


CALL_FN = {"anthropic": call_anthropic, "openai": call_openai, "groq": call_groq, "gemini": call_gemini}
# RPM limits per provider (used by rate limiter)
RPM_LIMITS = {"anthropic": 50, "openai": 60, "groq": GROQ_RPM, "gemini": GEMINI_RPM}
# Extra delay between stage 1 and stage 2 to avoid TPM window exhaustion
INTER_STAGE_DELAY = {"anthropic": 1, "openai": 1, "groq": GROQ_INTER_STAGE_DELAY, "gemini": 1}


# ---------------------------------------------------------------------------
# Three-stage generation
# ---------------------------------------------------------------------------

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


def analyze_style(call_fn, client, limiter: RateLimiter, abstract: str) -> str:
    print("    Stage 1: analyzing abstract style...", end=" ", flush=True)
    limiter.wait()
    result = call_fn(client, STAGE1_SYSTEM, STAGE1_USER.format(abstract=abstract))
    print("done.")
    return result


def generate_abstract(call_fn, client, limiter: RateLimiter, body: str, style_guide: str, n_words: int) -> str:
    print(f"    Stage 2: generating abstract (~{n_words} words)...", end=" ", flush=True)
    limiter.wait()
    text = call_fn(
        client,
        STAGE2_SYSTEM,
        STAGE2_USER.format(n_words=n_words, style_guide=style_guide, body=body),
    )
    # Hard-truncate to target word count to ensure length parity
    words = text.split()
    if len(words) > n_words + 15:
        text = " ".join(words[:n_words])
    print(f"done ({len(text.split())} words).")
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_done(pairs_csv: str) -> set:
    if not os.path.exists(pairs_csv):
        return set()
    return set(pd.read_csv(pairs_csv)["arxiv_id"].tolist())


def run(provider: str, papers_per_category: int, max_pairs: int, pairs_csv: str, pairs_csv_2: str):
    call_fn     = CALL_FN[provider]
    client      = build_client(provider)
    limiter     = RateLimiter(rpm=RPM_LIMITS[provider])
    stage_delay = INTER_STAGE_DELAY[provider]
    done        = load_done(pairs_csv) | load_done(pairs_csv_2)

    print(f"\nProvider  : {provider}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"File 1    : {pairs_csv} (first {max_pairs} pairs)")
    print(f"File 2    : {pairs_csv_2} (everything after)")
    print(f"Already done: {len(done)}")

    os.makedirs(os.path.dirname(pairs_csv), exist_ok=True)
    write_header_1 = not os.path.exists(pairs_csv)
    write_header_2 = not os.path.exists(pairs_csv_2)
    generated = 0
    skipped   = 0
    failed    = 0

    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"  CATEGORY: {category}  |  {generated} pairs collected so far")
        print(f"{'='*60}")
        papers = fetch_papers(category, papers_per_category)

        for i, paper in enumerate(papers):
            arxiv_id = paper.entry_id.split("/")[-1]
            print(f"\n  [{i+1}/{len(papers)}] {paper.title[:70]}")
            print(f"           arxiv:{arxiv_id}  published:{paper.published.date()}")

            if arxiv_id in done:
                print("    Already processed, skipping.")
                skipped += 1
                continue

            real_abstract = paper.summary.replace("\n", " ").strip()
            target_wc     = len(real_abstract.split())
            print(f"    Real abstract: {target_wc} words")

            body = download_body(paper)
            if body is None:
                failed += 1
                continue

            try:
                style_guide = analyze_style(call_fn, client, limiter, real_abstract)
                if stage_delay > 1:
                    print(f"    Waiting {stage_delay}s between stages to avoid TPM limit...", flush=True)
                    time.sleep(stage_delay)
                ai_abstract = generate_abstract(call_fn, client, limiter, body, style_guide, target_wc)
            except Exception as e:
                print(f"    LLM error: {e}")
                failed += 1
                continue

            record = pd.DataFrame([{
                "arxiv_id":      arxiv_id,
                "title":         paper.title,
                "category":      category,
                "published":     str(paper.published.date()),
                "real_abstract": real_abstract,
                "ai_abstract":   ai_abstract,
                "real_wc":       target_wc,
                "ai_wc":         len(ai_abstract.split()),
            }])

            if generated < max_pairs:
                record.to_csv(pairs_csv, mode="a", header=write_header_1, index=False)
                write_header_1 = False
                print(f"    Saved to file 1. [{generated+1}/{max_pairs}] | {limiter.stats()}")
            else:
                record.to_csv(pairs_csv_2, mode="a", header=write_header_2, index=False)
                write_header_2 = False
                print(f"    Saved to file 2. [{generated+1-max_pairs} extra] | {limiter.stats()}")

            generated += 1
            done.add(arxiv_id)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"  Generated : {generated}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Total pairs in CSV: {len(done)}")
    print(f"{'='*60}")


def merge(pairs_csv: str, out_csv: str):
    """Convert pairs CSV into a flat labeled dataset for training."""
    pairs = pd.read_csv(pairs_csv)
    print(f"\nMerging {len(pairs)} pairs into dataset...")

    human_rows = pairs[["real_abstract", "category", "published"]].copy()
    human_rows = human_rows.rename(columns={"real_abstract": "text"})
    human_rows["label"] = 0

    ai_rows = pairs[["ai_abstract", "category", "published"]].copy()
    ai_rows = ai_rows.rename(columns={"ai_abstract": "text"})
    ai_rows["label"] = 1

    combined = pd.concat([human_rows, ai_rows], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    combined.to_csv(out_csv, index=False)

    print(f"Saved {len(combined)} rows to {out_csv}")
    print(f"  Human : {(combined['label']==0).sum()}")
    print(f"  AI    : {(combined['label']==1).sum()}")

    combined["wc"] = combined["text"].str.split().str.len()
    print("\nWord count by label:")
    print(combined.groupby("label")["wc"].describe()[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "gemini"], default="gemini")
    parser.add_argument("--papers-per-category", type=int, default=250,
                        help="Papers to fetch per arXiv category (fetches more than needed to account for skips)")
    parser.add_argument("--max-pairs", type=int, default=1000,
                        help="Stop after generating this many pairs (default 1000)")
    parser.add_argument("--output", type=str, default="arxiv_pairs_1000.csv",
                        help="Output filename for pairs CSV (saved in data/raw/)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip generation and only rebuild dataset.csv from existing pairs")
    args = parser.parse_args()

    raw_dir  = os.path.join(os.path.dirname(__file__), "raw")
    proc_dir = os.path.join(os.path.dirname(__file__), "processed")

    pairs_csv   = os.path.join(raw_dir,  args.output)
    pairs_csv_2 = os.path.join(raw_dir,  args.output.replace(".csv", "_extra.csv"))
    out_csv     = os.path.join(proc_dir, args.output.replace("pairs", "dataset"))
    out_csv_2   = os.path.join(proc_dir, args.output.replace("pairs", "dataset").replace(".csv", "_extra.csv"))

    if not args.merge_only:
        run(args.provider, args.papers_per_category, args.max_pairs, pairs_csv, pairs_csv_2)

    if os.path.exists(pairs_csv):
        merge(pairs_csv, out_csv)
    if os.path.exists(pairs_csv_2):
        print("\n--- Extra pairs ---")
        merge(pairs_csv_2, out_csv_2)
    if not os.path.exists(pairs_csv) and not os.path.exists(pairs_csv_2):
        print("No pairs CSV found. Run without --merge-only first.")
