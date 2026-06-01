import argparse
import math
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"
ENDPOINT_COLUMNS = {"IC50": "pIC50", "KI": "pKi", "KD": "pKd"}


def endpoint_column(activity_type: str) -> str:
    activity_type = activity_type.upper()
    if activity_type not in ENDPOINT_COLUMNS:
        raise ValueError(f"Unsupported activity type: {activity_type}")
    return ENDPOINT_COLUMNS[activity_type]


def normalize_units(units: str) -> str:
    return (
        str(units or "").strip().replace("\u00b5", "U").replace("\u03bc", "U").upper()
    )


def activity_to_pvalue(value, units):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None

    scale = {"NM": 1e-9, "UM": 1e-6, "MM": 1e-3, "M": 1.0}.get(normalize_units(units))
    if scale is None:
        return None
    return -math.log10(value * scale)


def fetch_activities(
    chembl_target_id: str, activity_type: str = "IC50", page_size: int = 1000
) -> pd.DataFrame:
    activity_type = activity_type.upper()
    params = {
        "target_chembl_id": chembl_target_id,
        "standard_type": activity_type,
        "standard_relation": "=",
        "limit": page_size,
        "format": "json",
    }
    response = requests.get(BASE_URL, params=params | {"offset": 0}, timeout=60)
    response.raise_for_status()
    data = response.json()
    total = data.get("page_meta", {}).get("total_count", 0)
    pages = math.ceil(total / page_size) if total else 1

    frames = []
    for page in tqdm(
        range(pages), desc=f"Downloading {chembl_target_id} {activity_type}"
    ):
        page_response = requests.get(
            BASE_URL, params=params | {"offset": page * page_size}, timeout=60
        )
        page_response.raise_for_status()
        records = page_response.json().get("activities", [])
        if not records:
            continue

        df = pd.DataFrame.from_records(records)
        required = {
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "standard_relation",
        }
        if not required.issubset(df.columns):
            continue

        df = (
            df[
                [
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_value",
                    "standard_units",
                    "standard_type",
                    "standard_relation",
                ]
            ]
            .rename(
                columns={
                    "molecule_chembl_id": "molecule_chembl_id",
                    "canonical_smiles": "smiles",
                    "standard_value": "value",
                    "standard_units": "units",
                    "standard_type": "activity_type",
                    "standard_relation": "relation",
                }
            )
            .copy()
        )
        df["activity_type"] = df["activity_type"].astype(str).str.upper()
        df = df[
            (df["activity_type"] == activity_type)
            & (df["relation"] == "=")
            & df["smiles"].notna()
            & df["value"].notna()
            & df["units"].notna()
        ]
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "molecule_chembl_id",
                "smiles",
                "value",
                "units",
                "activity_type",
                "relation",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def prepare_endpoint_table(
    df: pd.DataFrame, chembl_id: str, activity_type: str
) -> pd.DataFrame:
    activity_type = activity_type.upper()
    y_col = endpoint_column(activity_type)
    df = df.copy()
    df[y_col] = [activity_to_pvalue(v, u) for v, u in zip(df["value"], df["units"])]
    df = df.dropna(subset=["smiles", y_col])
    df["endpoint"] = y_col
    df["target_chembl_id"] = chembl_id

    grouped = (
        df.groupby("smiles", as_index=False)
        .agg(
            **{
                y_col: (y_col, "median"),
                "n_measurements": (y_col, "size"),
                "target_chembl_id": ("target_chembl_id", "first"),
                "endpoint": ("endpoint", "first"),
            }
        )
        .sort_values("smiles")
        .reset_index(drop=True)
    )
    return grouped[["smiles", y_col, "target_chembl_id", "endpoint", "n_measurements"]]


def main():
    parser = argparse.ArgumentParser(
        description="Download one ChEMBL activity endpoint for one target."
    )
    parser.add_argument("--chembl_id", required=True, help="e.g. CHEMBL203")
    parser.add_argument(
        "--activity_type",
        default="IC50",
        choices=sorted(ENDPOINT_COLUMNS),
        help="Endpoint to download. Endpoints are not merged.",
    )
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--page_size", type=int, default=1000)
    args = parser.parse_args()

    df = fetch_activities(args.chembl_id, args.activity_type, args.page_size)
    if df.empty:
        print("No data.")
        return

    table = prepare_endpoint_table(df, args.chembl_id, args.activity_type)
    if table.empty:
        print("No usable activities after unit conversion.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"target_{args.chembl_id}_{args.activity_type.upper()}.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out} - {len(table)} compounds")


if __name__ == "__main__":
    main()
