import json
import pandas as pd

from pathlib import Path
from typing import List, Dict

def load_search_queries(filepath: Path) -> List[Dict[str, str | List[str]]]:
    with open(filepath, "r") as f:
        return json.load(f)


def write_results_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    print("Saving results to CSV...")

    if df.empty:
        print("No data to save.")
        return

    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"Saved to {filepath}")
