# Bangla E-commerce Review Normalization: Best-Version Implementation

This implementation combines:
- hybrid confidence-controlled Banglish-to-Bangla normalization
- sentiment-safe invariants
- product/entity protection
- safe Unicode/layout cleanup
- audit-friendly logging
- evaluation and manual audit tools

## Main idea

This is not simple "text cleaning." It is controlled surface-form harmonization for Bangla e-commerce sentiment analysis.

The final target is **Bangla-leaning mixed text**:
- Banglish sentiment words are normalized into Bangla script
- real English product/domain words are preserved
- product names, model IDs, and specifications are protected
- negation, intensifiers, punctuation emphasis, and emoji signals are not blindly removed

## Folder structure

```text
bangla_ecom_normalization_best/
├── normalizer_research.py
├── run_normalization.py
├── build_lexicon_candidates.py
├── make_audit_sample.py
├── evaluate_normalization.py
├── requirements.txt
├── resources/
│   ├── banglish_map.json
│   ├── dialect_map.json
│   ├── phrase_map.json
│   ├── keep_words.txt
│   └── english_words.txt
├── outputs/
└── audit/
```

## How to run

Put your dataset beside the scripts as:

```text
dataset_merged.csv
```

Then run:

```bash
python -m pip install -r requirements.txt
python run_normalization.py
```

Output:

```text
outputs/dataset_normalized_research.csv
```

The main training-ready column is:

```text
Review_norm_hybrid
```

## Output columns

- `Review_raw`: original review text
- `Review_norm_safe`: only safe cleanup applied
- `Review_norm_hybrid`: final hybrid normalized text
- `norm_language`: bangla / english / banglish / mixed
- `script_ratio_bn`: Bengali-script character ratio
- `script_ratio_latin`: Latin-script character ratio
- `norm_flags`: JSON flags for URL, emoji, elongation, digit policy, etc.
- `norm_actions`: JSON transformation log with rule IDs
- `norm_confidence`: average confidence for applied transformations
- `norm_version`: version string

## Build lexicon candidates

To extract Roman-script candidate words for manual review:

```bash
python build_lexicon_candidates.py --input dataset_merged.csv --text-col Review
```

Output:

```text
audit/roman_token_candidates.csv
```

Use this to expand `resources/banglish_map.json` safely.

## Create manual audit sample

After normalization:

```bash
python make_audit_sample.py --input outputs/dataset_normalized_research.csv --n 500
```

Output:

```text
audit/manual_audit_sample.csv
```

Suggested human judgment values:
- correct
- partial
- harmful_change
- missed_banglish
- product_token_damage
- negation_damage
- uncertain

## Evaluate normalization

```bash
python evaluate_normalization.py --input outputs/dataset_normalized_research.csv
```

Output:

```text
outputs/normalization_report.json
```

## Rule IDs

- R01_UNICODE_NFKC: Unicode compatibility normalization
- R02_BN_UNICODE: Bangla Unicode normalization if package is installed
- R03_URL: URL placeholder
- R04_EMAIL: email placeholder
- R05_PHONE: phone placeholder
- R06_HTML: HTML cleanup
- R07_DIGITS: Bengali digit to Western digit inside numeric spans
- R08_PUNCT_CAP: punctuation emphasis capped
- R09_PROTECT: product/entity span protection
- R10_PHRASE: phrase-level Banglish mapping
- R11_BANGLISH_LEX: curated Banglish lexicon mapping
- R12_DIALECT: dialect mapping
- R13_VARIANT: spelling variant mapping
- R14_SIMPLIFIED: simplified Roman spelling mapping
- R15_FUZZY: confidence-controlled fuzzy fallback

## Report description

You can describe this implementation like this:

> We implemented a hybrid confidence-controlled normalization pipeline for Bangla e-commerce sentiment reviews. The pipeline applies safe Unicode and layout normalization, protects product-domain spans such as brands, model IDs and specifications, and selectively normalizes high-confidence Banglish and dialectal forms into Bangla script. It preserves sentiment-critical signals such as negation, intensifiers, emoji and punctuation emphasis. Each transformation is logged with rule IDs and confidence scores, enabling auditability and intrinsic evaluation before downstream sentiment training.

## Resource Expansion v2

# Resource Expansion v2

This version improves the previous best package mainly through resource expansion and safer coverage.

Updated counts:
- Banglish map: 481 mappings
- Dialect map: 98 mappings
- Phrase map: 97 mappings
- Protected keep words: 1264 words
- English preserve words: 739 words

What improved:
- Added high-frequency review Banglish forms such as kaj, aro, vai/bhai, ata, gd, tnx/thnx-related handling.
- Added more e-commerce phrase mappings such as quality valo, delivery kharap, price beshi, dam kom.
- Added many English preserve words from high-frequency review text so genuine English is less likely to be altered.
- Added product/category tokens from product metadata into keep_words to better protect e-commerce terms.
- Kept the same research normalizer logic, because the main weakness was resource coverage rather than pipeline design.

Important:
Manual audit is still required. A larger lexicon improves coverage, but the final quality should be judged using the audit and evaluation scripts.


## Resource Expansion v3

# Resource Expansion v3: Maximum Safe Coverage

This version improves v2 with the largest safe expansion I can make without blindly corrupting training data.

Updated counts:
- Banglish map: 1453 mappings
- Dialect map: 285 mappings
- Phrase map: 124 mappings
- Protected keep words: 4238 words
- English preserve words: 740 words

Dataset-driven additions:
- Unique Roman review tokens scanned: 25452
- Unique product-name tokens scanned: 4206
- Unique category/source tokens scanned: 202
- Exported top Roman token review file: audit/top_roman_tokens_resource_review.csv

What improved:
1. More Banglish sentiment and review-action words were added.
2. More spelling variants were generated from trusted Banglish entries.
3. Product/category/source metadata tokens were added to protected keep words.
4. SKU-like alphanumeric review tokens were protected.
5. More genuine English review words were added to the preserve list.
6. More phrase-level mappings were added to reduce risky word-by-word changes.
7. Uncertain tokens were not force-mapped. They were exported for manual review instead.

Important safety note:
The goal is maximum safe coverage, not maximum random replacement.
If a token is uncertain, the normalizer keeps it unchanged. This is better for training than wrong normalization.
