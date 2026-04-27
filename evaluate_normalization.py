from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[A-Za-z]+|[\u0980-\u09FF]+|\d+(?:[./]\d+)?|[^\w\s]", flags=re.UNICODE)


def token_stats(series):
    tokens = []
    for text in series.fillna("").astype(str):
        tokens.extend(TOKEN_RE.findall(text.lower()))
    c = Counter(tokens)
    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(c),
        "hapax_tokens": sum(1 for _, v in c.items() if v == 1),
        "top_20": c.most_common(20),
    }


def script_stats(series):
    total_bn = total_lat = 0
    for text in series.fillna("").astype(str):
        total_bn += len(BANGLA_RE.findall(text))
        total_lat += len(LATIN_RE.findall(text))
    denom = max(total_bn + total_lat, 1)
    return {
        "bn_chars": total_bn,
        "latin_chars": total_lat,
        "ratio_bn": round(total_bn / denom, 4),
        "ratio_latin": round(total_lat / denom, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Normalized CSV")
    parser.add_argument("--output", default="outputs/normalization_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", low_memory=False)

    raw_col = "Review_raw"
    safe_col = "Review_norm_safe"
    hybrid_col = "Review_norm_hybrid"

    changed = (df[safe_col].astype(str) != df[hybrid_col].astype(str)).sum()
    raw_to_safe_changed = (df[raw_col].astype(str) != df[safe_col].astype(str)).sum()

    language_counts = df["norm_language"].value_counts(dropna=False).to_dict() if "norm_language" in df.columns else {}

    action_counter = Counter()
    total_actions = 0
    if "norm_actions" in df.columns:
        for item in df["norm_actions"].fillna("[]").astype(str):
            try:
                actions = json.loads(item)
                for a in actions:
                    action_counter[a.get("rule_id", "UNKNOWN")] += 1
                    total_actions += 1
            except Exception:
                continue

    report = {
        "rows": int(len(df)),
        "safe_cleanup_changed_rows": int(raw_to_safe_changed),
        "hybrid_changed_rows": int(changed),
        "hybrid_changed_percent": round(100 * changed / max(len(df), 1), 3),
        "total_logged_actions": int(total_actions),
        "language_counts": language_counts,
        "action_counts": dict(action_counter.most_common()),
        "raw_token_stats": token_stats(df[raw_col]) if raw_col in df.columns else {},
        "safe_token_stats": token_stats(df[safe_col]) if safe_col in df.columns else {},
        "hybrid_token_stats": token_stats(df[hybrid_col]) if hybrid_col in df.columns else {},
        "raw_script_stats": script_stats(df[raw_col]) if raw_col in df.columns else {},
        "hybrid_script_stats": script_stats(df[hybrid_col]) if hybrid_col in df.columns else {},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {args.output}")

    print("\nSummary")
    print(f"Rows: {report['rows']}")
    print(f"Safe cleanup changed rows: {report['safe_cleanup_changed_rows']}")
    print(f"Hybrid changed rows: {report['hybrid_changed_rows']} ({report['hybrid_changed_percent']}%)")
    print(f"Total logged actions: {report['total_logged_actions']}")


if __name__ == "__main__":
    main()
