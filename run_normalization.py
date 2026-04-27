from pathlib import Path
import pandas as pd
from normalizer_research import ResearchBanglaEcomNormalizer

INPUT_CSV = "dataset_merged.csv"
OUTPUT_CSV = "outputs/dataset_normalized_research.csv"
TEXT_COL = "Review"

df = pd.read_csv(INPUT_CSV, encoding="utf-8", low_memory=False)
normalizer = ResearchBanglaEcomNormalizer(resource_dir="resources", fuzzy_cutoff=0.91)

out = normalizer.normalize_dataframe(df, text_col=TEXT_COL)
Path("outputs").mkdir(exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Done. Saved: {OUTPUT_CSV}")
print(f"Rows: {len(out)}")
print("Use Review_norm_hybrid for the main training-ready normalized text.")
