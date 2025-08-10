import argparse, math
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"

def fetch_activities(chembl_target_id: str, page_size: int = 1000) -> pd.DataFrame:
    keep_types = {"IC50", "KI", "KD"}
    params = {
        "target_chembl_id": chembl_target_id,
        "standard_relation": "=",
        "limit": page_size,
        "format": "json",
    }
    r = requests.get(BASE_URL, params=params | {"offset": 0}, timeout=60)
    r.raise_for_status()
    data = r.json()
    total = data.get("page_meta", {}).get("total_count", 0)
    pages = math.ceil(total / page_size) if total else 1

    frames = []
    for p in tqdm(range(pages), desc=f"Downloading {chembl_target_id}"):
        rp = requests.get(BASE_URL, params=params | {"offset": p * page_size}, timeout=60)
        rp.raise_for_status()
        dp = rp.json()
        recs = dp.get("activities", [])
        if not recs:
            continue
        df = pd.DataFrame.from_records(recs)
        cols = ["canonical_smiles", "standard_value", "standard_units", "standard_type"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df.columns = ["smiles", "value", "units", "type"]
        df = df[df["type"].isin(keep_types)].dropna(subset=["smiles", "value", "units"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["smiles","value","units","type"])
    return pd.concat(frames, ignore_index=True)

def to_pIC50(row):
    v, u = row["value"], (row["units"] or "").strip().upper()
    try:
        v = float(v)
    except:
        return None
    if v <= 0:
        return None
    conv = {"NM": 1e-9, "UM": 1e-6, "MM": 1e-3, "M": 1.0}.get(u)
    if conv is None:
        return None
    import math
    return -math.log10(v * conv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_id", required=True, help="e.g. CHEMBL203")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    df = fetch_activities(args.chembl_id)
    if df.empty:
        print("No data.")
        return
    df["pIC50"] = df.apply(to_pIC50, axis=1)
    df = df.dropna(subset=["pIC50"]).drop_duplicates(subset=["smiles"])
    # Mediane par SMILES si multi mesures
    df = df.groupby("smiles", as_index=False)["pIC50"].median()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out = Path(args.outdir) / f"target_{args.chembl_id}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} — {len(df)} rows")

if __name__ == "__main__":
    main()
