"""
Turn raw_match/raw_original_labels.csv (from match_to_originals.py) into the final raw label list.

Accepted matches:
  A. normal      : distance <= 12 and margin >= 3      (already 'ok')
  B. duplicate   : distance <= 4  and margin == 0      (two identical originals; either one is the same picture)
  C. clear match : distance <= 20 and margin >= 5      (one clear best original, just more re-compression)
Everything else is listed in leftover.csv for a quick look.

Usage:  python finalize_raw_labels.py --match raw_match
Output: raw_labels_final.csv (original_path, raw_label, split, kind) and leftover.csv in the same folder
"""

import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--match", default="raw_match")
args = ap.parse_args()
m = Path(args.match).expanduser()
d = pd.read_csv(m / "raw_original_labels.csv")

A = (d.distance <= 12) & (d.margin >= 3)
B = (d.distance <= 4) & (d.margin == 0)
C = (d.distance <= 20) & (d.margin >= 5) & ~A
d["kind"] = "leftover"
d.loc[C, "kind"] = "clear"
d.loc[B, "kind"] = "duplicate"
d.loc[A, "kind"] = "normal"

final = d[d.kind != "leftover"]
final[["original_path", "raw_label", "split", "kind", "distance", "flat_path"]].to_csv(
    m / "raw_labels_final.csv", index=False
)
d[d.kind == "leftover"].to_csv(m / "leftover.csv", index=False)

total = len(d)
print(f"flat images matched to a candidate : {total}")
print(d.kind.value_counts().to_string())
print(f"\naccepted: {len(final)} of {total} ({100 * len(final) / total:.2f}%)")
print(f"raw classes in final list: {final.raw_label.nunique()}")
print(f"originals listed twice (identical duplicate pictures): {final.original_path.duplicated().sum()}")
if (d.kind == "leftover").any():
    print("\nleftover by distance (first rows):")
    print(
        d[d.kind == "leftover"]
        .groupby(["distance", "margin"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
