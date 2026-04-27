from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
ROMAN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="Review")
    parser.add_argument("--output", default="audit/roman_token_candidates.csv")
    parser.add_argument("--min-freq", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", low_memory=False)
    counter = Counter()

    for text in df[args.text_col].fillna("").astype(str):
        for tok in ROMAN_TOKEN_RE.findall(text.lower()):
            counter[tok] += 1

    rows = []
    for token, freq in counter.most_common():
        if freq >= args.min_freq:
            rows.append({"token": token, "freq": freq, "decision": "", "target_bn": "", "notes": ""})

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved candidates: {args.output}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
