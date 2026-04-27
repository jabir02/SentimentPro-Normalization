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
