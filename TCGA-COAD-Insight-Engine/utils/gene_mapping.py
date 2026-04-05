import pandas as pd
import os

_MAPPING_CACHE = None

def _load_mapping() -> dict:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        return _MAPPING_CACHE
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "ensembl_to_symbol.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=str)
        _MAPPING_CACHE = dict(zip(df["ensembl_id"], df["symbol"]))
    else:
        _MAPPING_CACHE = {}
    return _MAPPING_CACHE


def strip_version(ensembl_id: str) -> str:
    return ensembl_id.split(".")[0] if "." in ensembl_id else ensembl_id


def map_ensembl_to_symbol(ensembl_ids: list[str]) -> dict[str, str]:
    mapping = _load_mapping()
    result = {}
    for eid in ensembl_ids:
        base = strip_version(eid)
        result[eid] = mapping.get(base, base)
    return result
