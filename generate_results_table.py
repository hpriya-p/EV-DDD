import json
import glob
import re
import pandas as pd
from pathlib import Path

results_dir = Path("results")

# Collect all experiment IDs
heuristic_files = {}
default_files = {}

for path in results_dir.glob("experiment_*-*-tractor.json"):
    match = re.match(r"experiment_([a-f0-9]+)-(heuristic|default)-tractor\.json", path.name)
    if match:
        exp_id, config = match.group(1), match.group(2)
        if config == "heuristic":
            heuristic_files[exp_id] = path
        elif config == "default":
            default_files[exp_id] = path

all_ids = sorted(heuristic_files.keys() & default_files.keys())

rows = []
for exp_id in all_ids:
    row = {"id": exp_id}

    for config, files in [("heuristic", heuristic_files), ("default", default_files)]:
        if exp_id in files:
            with open(files[exp_id]) as f:
                data = json.load(f)
            row[f"{config} cost"] = data.get("objective")
            size = data.get("properties", {}).get("size_of_graph")
            row[f"{config} size"] = tuple(size) if size is not None else None
            row[f"{config} iter"] = data.get("properties", {}).get("n_iterations")
        else:
            row[f"{config} cost"] = None
            row[f"{config} size"] = None
            row[f"{config} iter"] = None
    rows.append(row)

df = pd.DataFrame(rows, columns=[
    "id",
    "heuristic cost",
    "default cost",
    "heuristic size",
    "default size",
    "heuristic iter",
    "default iter",
])

print(df.to_string(index=False))
df.to_csv("results/results_table.csv", index=False)
print("\nSaved to results/results_table.csv")
