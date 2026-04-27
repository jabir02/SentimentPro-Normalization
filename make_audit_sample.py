from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Normalized CSV")
    parser.add_argument("--output", default="audit/manual_audit_sample.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sentiment-col", default="Sentiment")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", low_memory=False)

    # Prefer rows that changed, but include unchanged rows too.
    changed = df[df["Review_norm_safe"].astype(str) != df["Review_norm_hybrid"].astype(str)].copy()
    unchanged = df[df["Review_norm_safe"].astype(str) == df["Review_norm_hybrid"].astype(str)].copy()

    samples = []
    half = args.n // 2

    if args.sentiment_col in df.columns:
        for source_df, take_n in [(changed, half), (unchanged, args.n - half)]:
            if len(source_df) == 0:
                continue
            per_group = max(1, take_n // max(source_df[args.sentiment_col].nunique(), 1))
            sampled = (
                source_df.groupby(args.sentiment_col, group_keys=False)
                .apply(lambda x: x.sample(min(len(x), per_group), random_state=args.random_state))
            )
            samples.append(sampled)
    else:
        if len(changed):
            samples.append(changed.sample(min(len(changed), half), random_state=args.random_state))
        if len(unchanged):
            samples.append(unchanged.sample(min(len(unchanged), args.n - half), random_state=args.random_state))

    audit = pd.concat(samples, ignore_index=True).head(args.n) if samples else df.sample(min(len(df), args.n), random_state=args.random_state)

    cols = [c for c in [
        "Rating", "Product Name", "Product Category", "Sentiment", "Emotion",
        "Review_raw", "Review_norm_safe", "Review_norm_hybrid",
        "norm_language", "norm_actions", "norm_flags"
    ] if c in audit.columns]
    audit = audit[cols].copy()

    audit["human_judgment"] = ""
    audit["error_type"] = ""
    audit["human_notes"] = ""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved audit sample: {args.output}")
    print("Suggested human_judgment values: correct, partial, harmful_change, missed_banglish, product_token_damage, negation_damage, uncertain")


if __name__ == "__main__":
    main()
