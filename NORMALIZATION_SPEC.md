# Normalization Specification

## Core principle

Treat normalization as controlled surface-form harmonization under sentiment-safe invariants, not as generic cleaning.

## Invariants

The pipeline must not intentionally remove or corrupt:
- negation: না, নাই, নেই, no, not
- intensifiers: খুব, অনেক, khub, onek, very
- product entities: brand names, model IDs, SKUs, specifications
- punctuation emphasis unless capped safely
- emoji and emoticons without explicit policy
- real English domain terms such as camera, battery, delivery, seller

## Target output

The target is Bangla-leaning mixed text:
- Banglish sentiment words become Bangla script
- real English product/domain words stay unchanged
- product/model/spec tokens stay intact
- low-confidence uncertain forms are left unchanged

## Evaluation

Intrinsic:
- changed row count
- token/type count reduction
- script ratio shift
- logged action counts
- protected span count

Manual:
- 500-review audit sample
- categories: correct, partial, harmful change, missed Banglish, product token damage, negation damage, uncertain

Downstream:
- compare raw vs safe vs hybrid normalization in the training pipeline when possible