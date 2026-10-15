"""
Converts copy-pasted solver output into valid Python assignments and
injects them into plot_tri_city.py (lines 12-16).

Usage:
    python3 fix_solution.py < pasted_output.txt
    python3 fix_solution.py pasted_output.txt
    python3 fix_solution.py          # then paste, finish with Ctrl+D

The script expects lines of the form printed by the solver:
    x_load: {...}
    x_ener: {...}
    a: {...}
    n: {...}
    Pos:{0: ...}          (optional)
    {0: ...}              (bare pos dict, also accepted)

It outputs corrected Python lines and overwrites the data block in
plot_tri_city.py (lines 12-16).
"""

import re
import sys

PLOT_FILE = 'plot_tri_city.py'

def fix_line(line):
    # Replace 'np.int64(X)' strings with bare integers
    line = re.sub(r"'np\.int64\((\d+)\)'", r'\1', line)
    # Fix assignment syntax
    line = re.sub(r'^(x_load|x_ener|a|n): ', r'\1 = ', line)
    # Fix bare pos dict or 'Pos:{...}'
    line = re.sub(r'^[Pp]os\s*[:=]\s*\{', 'pos_latlon = {', line)
    if re.match(r'^\{0:', line):
        line = 'pos_latlon = ' + line
    return line

def parse_input(raw):
    """Return a dict of variable name -> fixed assignment line."""
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fixed = fix_line(line)
        for var in ('x_load', 'x_ener', 'a', 'n', 'pos_latlon'):
            if fixed.startswith(var + ' =') or fixed.startswith(var + '='):
                result[var] = fixed
                break
    return result

def inject_into_plot(assignments):
    """Overwrite lines 12-16 (1-indexed) of plot_tri_city.py."""
    with open(PLOT_FILE, 'r') as f:
        lines = f.readlines()

    # Build replacement block (lines 12-16 are indices 11-15)
    block = []
    for var in ('x_load', 'x_ener', 'a', 'n', 'pos_latlon'):
        if var in assignments:
            block.append(assignments[var] + '\n')
        else:
            # Keep existing line if not provided
            idx = {'x_load': 11, 'x_ener': 12, 'a': 13, 'n': 14, 'pos_latlon': 15}[var]
            if idx < len(lines):
                block.append(fix_line(lines[idx]))
            else:
                block.append(f'{var} = {{}}\n')

    lines[11:16] = block

    with open(PLOT_FILE, 'w') as f:
        f.writelines(lines)
    print(f"Updated {PLOT_FILE}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            raw = f.read()
    else:
        print("Paste solver output, then press Ctrl+D:")
        raw = sys.stdin.read()

    assignments = parse_input(raw)

    if not assignments:
        print("No recognised variables found in input.", file=sys.stderr)
        sys.exit(1)

    print("Parsed variables:", list(assignments.keys()))
    inject_into_plot(assignments)
