import pandas as pd

PATH = "data/processed/spy_features_w_targets.csv"

df = pd.read_csv(PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Spot case-insensitive duplicates
lower_map = {}
dupes = []
for c in df.columns:
    k = c.lower()
    if k in lower_map:
        dupes.append((lower_map[k], c))
    else:
        lower_map[k] = c

print("\nCase-insensitive duplicates:")
for a, b in dupes:
    print(f" - {a}  vs  {b}")

# Quick peek
print("\nHead:")
print(df.head(3))