"""
Generate AI essays using LLM APIs, then merge with human essays to produce a labeled dataset.

Usage:
    python generate_ai_essays.py --provider openai --limit 500
    python generate_ai_essays.py --provider anthropic --limit 500
    python generate_ai_essays.py --provider groq --limit 500

Set the appropriate environment variable for your chosen provider:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY

Output:
    data/raw/ai_essays.csv       - generated AI essays with label=1
    data/processed/dataset.csv   - merged human + AI essays with label column
"""

import argparse
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

HUMAN_CSV = os.path.join(os.path.dirname(__file__), "raw", "essays_sorted.csv")
AI_CSV    = os.path.join(os.path.dirname(__file__), "raw", "ai_essays.csv")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "processed", "dataset.csv")

SYSTEM_PROMPT = (
    "You are an essay writer. Write a thoughtful, well-structured essay on the given topic. "
    "Use clear paragraphs. Do not include a title. Aim for 4-6 paragraphs."
)

def build_user_prompt(title: str, description: str) -> str:
    return f"Topic: {title}\n\nPrompt: {description}\n\nWrite the essay now."


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

def call_openai(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def call_anthropic(client, prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def call_groq(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_client(provider: str):
    if provider == "openai":
        import openai
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "groq":
        from groq import Groq
        return Groq(api_key=os.environ["GROQ_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


CALL_FN = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "groq":      call_groq,
}


def load_existing(path: str) -> set:
    """Return set of titles already generated (for resuming interrupted runs)."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["title"].tolist())


def generate_ai_essays(provider: str, limit: int, delay: float):
    human_df = pd.read_csv(HUMAN_CSV)
    human_df = human_df.dropna(subset=["description", "essay", "title"])

    if limit:
        human_df = human_df.head(limit)

    client  = build_client(provider)
    call_fn = CALL_FN[provider]

    already_done = load_existing(AI_CSV)
    print(f"Loaded {len(human_df)} human essays. {len(already_done)} AI essays already generated.")

    os.makedirs(os.path.dirname(AI_CSV), exist_ok=True)
    write_header = not os.path.exists(AI_CSV)

    generated = 0
    errors    = 0

    for _, row in human_df.iterrows():
        title       = str(row["title"]).strip()
        description = str(row["description"]).strip()

        if title in already_done:
            continue

        prompt = build_user_prompt(title, description)

        try:
            ai_text = call_fn(client, prompt)
        except Exception as e:
            print(f"  ERROR on '{title}': {e}")
            errors += 1
            time.sleep(delay * 2)
            continue

        record = pd.DataFrame([{
            "title":       title,
            "description": description,
            "essay":       ai_text,
            "provider":    provider,
            "label":       1,
        }])

        record.to_csv(AI_CSV, mode="a", header=write_header, index=False)
        write_header = False
        generated   += 1

        print(f"  [{generated}] Generated essay for: {title[:60]}")
        time.sleep(delay)

    print(f"\nDone. Generated {generated} essays, {errors} errors.")


def merge_datasets():
    human_df = pd.read_csv(HUMAN_CSV)
    human_df = human_df[["title", "description", "essay"]].copy()
    human_df["label"]    = 0
    human_df["provider"] = "human"

    ai_df = pd.read_csv(AI_CSV)
    ai_df = ai_df[["title", "description", "essay", "label", "provider"]].copy()

    combined = pd.concat([human_df, ai_df], ignore_index=True)
    combined = combined.rename(columns={"essay": "text"})
    combined = combined.dropna(subset=["text"])
    combined = combined[combined["text"].str.strip().str.len() > 100]
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nMerged dataset saved to {OUT_CSV}")
    print(f"  Human essays : {(combined['label'] == 0).sum()}")
    print(f"  AI essays    : {(combined['label'] == 1).sum()}")
    print(f"  Total        : {len(combined)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI essays and build labeled dataset.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "groq"], default="anthropic",
                        help="LLM provider to use for generation.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of essays to generate (0 = all).")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between API calls to avoid rate limits.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip generation and only merge existing ai_essays.csv with human data.")
    args = parser.parse_args()

    if not args.merge_only:
        generate_ai_essays(args.provider, args.limit, args.delay)

    if os.path.exists(AI_CSV):
        merge_datasets()
    else:
        print("No ai_essays.csv found. Run without --merge-only first.")
