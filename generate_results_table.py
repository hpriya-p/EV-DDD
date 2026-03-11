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

all_ids = sorted(heuristic_files.keys() | default_files.keys())

col_names = {
    "heuristic": ("Heuristic obj. value", "Network size (heuristic)", "# iterations (heuristic)"),
    "default":   ("Optimal value",         "Network size (exact)",     "# iterations (exact)"),
}
columns = [
    "id",
    "Heuristic obj. value",
    "Optimal value",
    "Network size (heuristic)",
    "Network size (exact)",
    "# iterations (heuristic)",
    "# iterations (exact)",
]

rows = []
for exp_id in all_ids:
    row = {"id": exp_id, "n": None}
    for config, files in [("heuristic", heuristic_files), ("default", default_files)]:
        cost_col, size_col, iter_col = col_names[config]
        if exp_id in files:
            with open(files[exp_id]) as f:
                data = json.load(f)
            if row["n"] is None:
                row["n"] = data.get("n")
            row[cost_col] = data.get("objective")
            size = data.get("properties", {}).get("size_of_graph")
            row[size_col] = tuple(size) if size is not None else None
            row[iter_col] = data.get("properties", {}).get("n_iterations")
        else:
            row[cost_col] = None
            row[size_col] = None
            row[iter_col] = None
    rows.append(row)

df_all = pd.DataFrame(rows, columns=["n"] + columns)

for n_val, group in sorted(df_all.groupby("n")):
    df = group[columns].sort_values("Heuristic obj. value").reset_index(drop=True)
    print(f"\n=== n = {n_val} ===")
    print(df.to_string(index=False, na_rep="---"))
    csv_path = f"results/results_table_n{n_val}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    print(f"\n--- LaTeX (n={n_val}) ---")
    print(df.to_latex(index=False, na_rep="---"))
