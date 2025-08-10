# scripts/fetch_chembl.py
import os, sys, math
from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client

# ensure we can import core from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import TARGETS  # keep in sync with the app

UNIT = {'M': 1.0, 'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9}

def write(gene: str, limit: int = 2000):
    tcli = new_client.target
    acli = new_client.activity
    hits = tcli.filter(target_synonym__icontains=gene) or tcli.search(gene)
    if not hits:
        print(f"[warn] no target for {gene}"); return
    tid = hits[0]['target_chembl_id']

    acts = acli.filter(
        target_chembl_id=tid,
        standard_relation="=",
        standard_type__in=["IC50","Ki","Kd"]
    ).only("canonical_smiles","standard_value","standard_units")[:limit]

    rows = []
    for r in acts:
        smi = r.get("canonical_smiles")
        val = r.get("standard_value")
        u   = r.get("standard_units")
        if not (smi and val and u in UNIT):
            continue
        try:
            pic50 = -math.log10(float(val) * UNIT[u])
        except Exception:
            continue
        rows.append([smi, pic50])

    if not rows:
        print(f"[warn] no clean rows for {gene}"); return

    df = (pd.DataFrame(rows, columns=["smiles","pIC50"])
            .groupby("smiles", as_index=False)["pIC50"].median())
    out = Path("data") / f"ligands_{gene}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out} ({len(df)} rows)")

if __name__ == "__main__":
    for g in TARGETS:
        write(g)
    print("[done]")
