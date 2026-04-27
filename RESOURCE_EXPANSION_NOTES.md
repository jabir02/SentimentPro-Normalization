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
