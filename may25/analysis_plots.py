import glob, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import re, ast
from collections import defaultdict
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Parse result files ─────────────────────────────────────────────────────────
# Filename convention: results_<date>_<mode>_<Demand><Battery><Cost>.txt
# Ignore date and last letter (cost).  H=High  M=Medium  L=Low

def _parse_filename(filepath):
    m = re.search(r'results_\d+_(\w+)_([LMH])([LMH])[LMH]\.txt$',
                  os.path.basename(filepath))
    if m:
        return m.group(1), m.group(2), m.group(3)   # mode, demand, battery
    return None, None, None


def _parse_file(filepath):
    """Return dict with obj, a, n, node_coords parsed from a result file."""
    with open(filepath) as f:
        lines = [ln.rstrip('\n') for ln in f]

    parsed = {}
    for line in lines:
        s = line.strip()

        if s.startswith('Obj Value:'):
            parsed['obj'] = float(s.split('Obj Value:')[1].strip())

        elif s.startswith('a:'):
            parsed['a'] = ast.literal_eval(s[2:].strip())

        elif s.startswith('n:'):
            parsed['n'] = ast.literal_eval(s[2:].strip())

        # Node-coord line: bare dict {int: (lat, lon), ...}, no np.int64
        elif s.startswith('{') and ': (' in s and 'np.int64' not in s:
            try:
                parsed['node_coords'] = ast.literal_eval(s)
            except Exception:
                pass

    return parsed


records     = {}
node_coords = None

for fp in sorted(glob.glob('may25/results_*.txt')):
    mode, demand, battery = _parse_filename(fp)
    if mode is None:
        continue
    p   = _parse_file(fp)
    key = f'{mode}_{demand}{battery}M'
    records[key] = dict(
        mode=mode, demand=demand, battery=battery,
        obj=p.get('obj', float('nan')),
        a=p.get('a', {}),
        n=p.get('n', {}),
        filepath=fp,
    )
    if node_coords is None:
        node_coords = p.get('node_coords')

print(f'Loaded {len(records)} result files: {sorted(records)}')
assert node_coords, 'Could not parse node coordinates from any result file'

PROJ         = ccrs.PlateCarree()
MAP_EXTENT   = [-123, -93, 29.5, 40]   # [lon_min, lon_max, lat_min, lat_max]
MODE_COLOR   = {'tractor': '#1565C0', 'charger': '#B71C1C'}
MODE_MARKER  = {'tractor': 'o',       'charger': 's'}


# ── Helpers ────────────────────────────────────────────────────────────────────

def cost_for(mode, demand, battery):
    key = f'{mode}_{demand}{battery}M'
    return records[key]['obj'] if key in records else None


def setup_map(ax, title=''):
    ax.set_extent(MAP_EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor='#f5f0e8', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor='#d6eaf8', zorder=0)
    ax.add_feature(cfeature.STATES,    edgecolor='#888888',
                   linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#555555',
                   linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.BORDERS,   edgecolor='#555555',
                   linewidth=0.8, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    # Plot all network nodes as light background dots
    for lat, lon in node_coords.values():
        ax.plot(lon, lat, '.', color='#bbbbbb', markersize=3.5,
                transform=PROJ, zorder=2)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)


def label_node(ax, lon, lat, text, color='black', fontsize=7):
    ax.text(lon + 0.35, lat + 0.25, text,
            fontsize=fontsize, color=color, fontweight='bold',
            transform=PROJ, zorder=6,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      alpha=0.6, edgecolor='none'))


DEMAND_ORDER  = ['L', 'M', 'H']
DEMAND_LABELS = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

CHARGER_CASES = [
    ('charger_LMM', '#2E7D32', 'o', 'Low demand',    0.0, -0.55),
    ('charger_MMM', '#1565C0', 's', 'Medium demand',  0.0,  0.0 ),
    ('charger_HMM', '#C62828', '^', 'High demand',    0.0, +0.55),
]
TRACTOR_CASES = [
    ('tractor_LMM', '#2E7D32', 'o', 'Low demand',    0.0, -0.55),
    ('tractor_MMM', '#1565C0', 's', 'Medium demand',  0.0,  0.0 ),
    ('tractor_HMM', '#C62828', '^', 'High demand',    0.0, +0.55),
]


def save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── Q1 — Cost vs Demand ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))

for mode in ('tractor', 'charger'):
    xs, ys = [], []
    for d in DEMAND_ORDER:
        c = cost_for(mode, d, 'M')
        if c is not None:
            xs.append(DEMAND_LABELS[d])
            ys.append(c)
    ax1.plot(xs, ys, color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
             linewidth=2.2, markersize=9, label=mode.capitalize())

ax1.set_xlabel('Demand Level', fontsize=11)
ax1.set_ylabel('Average Cost', fontsize=11)
ax1.set_title('Cost vs. Demand', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e3:.1f}k'))
ax1.set_ylim(bottom=0)
save(fig1, 'may25/q1_cost_vs_demand.png')

# ── Q2 — Cost vs Battery Capacity ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))

for mode in ('tractor', 'charger'):
    xs, ys = [], []
    for b in ['M', 'H']:
        c = cost_for(mode, 'M', b)
        if c is not None:
            xs.append({'M': 'Medium', 'H': 'High'}[b])
            ys.append(c)
    if xs:
        ax2.plot(xs, ys, color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
                 linewidth=2.2, markersize=9, label=mode.capitalize())

ax2.set_xlabel('Battery Capacity', fontsize=11)
ax2.set_ylabel('Average Cost', fontsize=11)
ax2.set_title('Cost vs. Battery Capacity',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e3:.1f}k'))
ax2.set_ylim(bottom=0)
save(fig2, 'may25/q2_cost_vs_battery.png')

# ── Q3 — Charger placement map ────────────────────────────────────────────────
DEMAND_COLOR = {'L': '#2E7D32', 'M': '#1565C0', 'H': '#C62828'}
DEMAND_NAME  = {'L': 'Low demand', 'M': 'Medium demand', 'H': 'High demand'}
PIE_R = 0.25  # radius in degrees
PIE_OFFSETS = {7: (-0.55, 0.0), 11: (0.55, 0.0)}  # (dlon, dlat) nudges for crowded nodes

# Build {node_id: [colors]} in L→M→H order
node_demands = defaultdict(list)
for demand in ['L', 'M', 'H']:
    key = f'charger_{demand}MM'
    for nid in records.get(key, {}).get('a', {}):
        node_demands[nid].append(DEMAND_COLOR[demand])

fig3, ax3 = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=PROJ))
setup_map(ax3)

for nid, colors in node_demands.items():
    lat, lon = node_coords[nid]
    dlon, dlat = PIE_OFFSETS.get(nid, (0.0, 0.0))
    lon, lat = lon + dlon, lat + dlat
    r = PIE_R
    n = len(colors)
    sector = 360 / n
    for i, color in enumerate(colors):
        wedge = mpatches.Wedge(
            (lon, lat), r,
            90 + i * sector, 90 + (i + 1) * sector,
            facecolor=color, edgecolor='white',
            linewidth=0.6, alpha=0.88,
            transform=PROJ, zorder=4)
        ax3.add_patch(wedge)
    ax3.add_patch(mpatches.Circle(
        (lon, lat), r,
        facecolor='none', edgecolor='#333333',
        linewidth=0.7, transform=PROJ, zorder=5))

demand_handles = [mpatches.Patch(color=DEMAND_COLOR[d], label=DEMAND_NAME[d])
                  for d in ['L', 'M', 'H']]
ax3.legend(handles=demand_handles, loc='lower left', fontsize=9, framealpha=0.9)
save(fig3, 'may25/q3_charger_placement.png')

BAT_COLOR = {'M': '#1565C0', 'H': '#C62828'}
BAT_NAME  = {'M': 'Regular battery capacity', 'H': 'Increased battery capacity'}


def _draw_count_diffs(ax, node_color_map, case_counts, ref_key,
                      short_names, colors=None, offsets=None,
                      node_label_offsets=None):
    """
    For each node where any case count differs from the reference case,
    draw color-coded labels (one per case, stacked vertically).
    node_label_offsets: {nid: (extra_dlon, extra_dlat)} to nudge specific labels.
    """
    _offsets = PIE_OFFSETS.copy()
    if offsets:
        _offsets.update(offsets)
    _colors = colors or {}
    _nloff = node_label_offsets or {}
    ref = case_counts[ref_key]
    for nid in sorted(node_color_map):
        lat, lon = node_coords[nid]
        dlon, dlat = _offsets.get(nid, (0.0, 0.0))
        lon, lat = lon + dlon, lat + dlat
        extra_dlon, extra_dlat = _nloff.get(nid, (0.0, 0.0))
        row = 0
        for key, counts in case_counts.items():
            if key == ref_key:
                continue
            diff = counts.get(nid, 0) - ref.get(nid, 0)
            if diff != 0:
                color = _colors.get(key, '#333333')
                ax.text(lon + 0.3 + extra_dlon,
                        lat - PIE_R - 0.1 - row * 0.28 + extra_dlat,
                        f'{short_names[key]}:{int(diff):+d}',
                        fontsize=5.5, color=color, fontweight='bold',
                        transform=PROJ, zorder=8, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  alpha=0.7, edgecolor='none'))
                row += 1


def _draw_pie_nodes(ax, node_color_map, show_ids=False, offsets=None):
    """Draw pie-chart circles (radius PIE_R) for a {node_id: [colors]} dict."""
    _offsets = PIE_OFFSETS.copy()
    if offsets:
        _offsets.update(offsets)
    for nid, colors in node_color_map.items():
        lat, lon = node_coords[nid]
        dlon, dlat = _offsets.get(nid, (0.0, 0.0))
        lon, lat = lon + dlon, lat + dlat
        n = len(colors)
        sector = 360 / n
        for i, color in enumerate(colors):
            ax.add_patch(mpatches.Wedge(
                (lon, lat), PIE_R,
                90 + i * sector, 90 + (i + 1) * sector,
                facecolor=color, edgecolor='white',
                linewidth=0.6, alpha=0.88,
                transform=PROJ, zorder=4))
        ax.add_patch(mpatches.Circle(
            (lon, lat), PIE_R,
            facecolor='none', edgecolor='#333333',
            linewidth=0.7, transform=PROJ, zorder=5))
        if show_ids:
            label_node(ax, lon, lat, str(nid), color='#333333')


# ── Q4 — Tractor placement map (varying demand) ───────────────────────────────
node_demands_t = defaultdict(list)
for demand in ['L', 'M', 'H']:
    for nid in records.get(f'tractor_{demand}MM', {}).get('n', {}):
        node_demands_t[nid].append(DEMAND_COLOR[demand])

q4_offsets = {89: (0.0, -0.55), 85: (0.0, -0.55), 105: (0.55, 0.0)}
q4_counts  = {d: records.get(f'tractor_{d}MM', {}).get('n', {}) for d in ['L','M','H']}

fig4, ax4 = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=PROJ))
setup_map(ax4)
_draw_pie_nodes(ax4, node_demands_t, offsets=q4_offsets)
_draw_count_diffs(ax4, node_demands_t, q4_counts, ref_key='L',
                  short_names={'M': 'M', 'H': 'H'},
                  colors={'M': DEMAND_COLOR['M'], 'H': DEMAND_COLOR['H']},
                  offsets=q4_offsets,
                  node_label_offsets={62: (1.0, 0.0), 98: (0.8, 0.0)})
ax4.legend(handles=[mpatches.Patch(color=DEMAND_COLOR[d], label=DEMAND_NAME[d])
                    for d in ['L', 'M', 'H']],
           loc='lower left', fontsize=9, framealpha=0.9)
save(fig4, 'may25/q4_tractor_placement.png')

# ── Q3b — Charger placement map (varying battery) ─────────────────────────────
node_bat_c = defaultdict(list)
for bat in ['M', 'H']:
    for nid in records.get(f'charger_M{bat}M', {}).get('a', {}):
        node_bat_c[nid].append(BAT_COLOR[bat])

fig3b, ax3b = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=PROJ))
setup_map(ax3b)
_draw_pie_nodes(ax3b, node_bat_c, show_ids=False)
ax3b.legend(handles=[mpatches.Patch(color=BAT_COLOR[b], label=BAT_NAME[b])
                     for b in ['M', 'H']],
            loc='lower left', fontsize=9, framealpha=0.9)
save(fig3b, 'may25/q3b_charger_placement_battery.png')

# ── Q4b — Tractor placement map (varying battery) ─────────────────────────────
node_bat_t = defaultdict(list)
for bat in ['M', 'H']:
    for nid in records.get(f'tractor_M{bat}M', {}).get('n', {}):
        node_bat_t[nid].append(BAT_COLOR[bat])

q4b_offsets = {89: (0.0, -0.55), 105: (0.55, 0.0)}
q4b_counts  = {b: records.get(f'tractor_M{b}M', {}).get('n', {}) for b in ['M','H']}

fig4b, ax4b = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=PROJ))
setup_map(ax4b)
_draw_pie_nodes(ax4b, node_bat_t, offsets=q4b_offsets)
_draw_count_diffs(ax4b, node_bat_t, q4b_counts, ref_key='H',
                  short_names={'M': 'Δ'},
                  colors={'M': BAT_COLOR['M']},
                  offsets=q4b_offsets)
ax4b.legend(handles=[mpatches.Patch(color=BAT_COLOR[b], label=BAT_NAME[b])
                     for b in ['M', 'H']],
            loc='lower left', fontsize=9, framealpha=0.9,
            title='Battery capacity', title_fontsize=9)
save(fig4b, 'may25/q4b_tractor_placement_battery.png')

# ── Q5 / Q6 — Flow maps ────────────────────────────────────────────────────────
# solution[k] = [load_flow_dict, energy_flow_dict]
# Load (index 0): truck movements.  Energy (index 1): energy carried/transferred.

LOAD_COLOR = '#1565C0'

OD_CMAP = cm.tab10

# Existing real-world charging stations in the network (pre-computed from stations CSV)
EXISTING_STATION_NODES = {3, 8, 18, 37, 42, 44, 47, 48, 52, 57, 65, 70, 77, 81, 82, 83, 87, 100}

# City labels for all potential origin/destination nodes across both models
CITY_LABELS = {56: 'PHX', 69: 'SF', 90: 'DAL', 91: 'LA',
               89: 'AMA', 43: 'ELK'}   # charger-model endpoints


def make_od_labels(od_nodes):
    """Build {od_key: 'Origin → Destination'} dynamically from parsed od_nodes."""
    labels = {}
    for k, (origin, dest) in od_nodes.items():
        o_name = CITY_LABELS.get(origin, f'Node {origin}')
        d_name = CITY_LABELS.get(dest,   f'Node {dest}')
        labels[k] = f'{o_name} → {d_name}'
    return labels


_SOLUTION_OVERRIDES = {}   # {filepath: solution_dict} — bypasses file reading


def _parse_solution(filepath):
    if filepath in _SOLUTION_OVERRIDES:
        return _SOLUTION_OVERRIDES[filepath]
    with open(filepath) as f:
        lines = f.readlines()
    sol_line = ''
    for line in reversed(lines):
        s = line.strip()
        if s.startswith('{') and 'np.int64' in s:
            sol_line = s
            break
    if not sol_line:
        return {}
    sol_line = re.sub(r'np\.int64\((\d+)\)', r'\1', sol_line)
    return ast.literal_eval(sol_line)


def parse_load_flows(filepath):
    """Aggregate truck (load) flows across all OD pairs → {(n1,n2): flow}."""
    solution = _parse_solution(filepath)
    geo = defaultdict(float)
    for route_list in solution.values():
        for edge, flow in route_list[0].items():
            n1, n2 = edge[0][0], edge[1][0]
            if n1 != n2 and n1 in node_coords and n2 in node_coords:
                geo[(n1, n2)] += flow
    return dict(geo)


def parse_energy_only_flows(filepath):
    """
    Edges that carry energy flow but have NO corresponding load flow on the
    same geographic arc → {(n1,n2): ener_flow}.
    """
    solution = _parse_solution(filepath)
    load_geo = defaultdict(float)
    ener_geo = defaultdict(float)
    for route_list in solution.values():
        for edge, flow in route_list[0].items():   # load
            n1, n2 = edge[0][0], edge[1][0]
            if n1 != n2 and n1 in node_coords and n2 in node_coords:
                load_geo[(n1, n2)] += flow
        for edge, flow in route_list[1].items():   # energy
            n1, n2 = edge[0][0], edge[1][0]
            if n1 != n2 and n1 in node_coords and n2 in node_coords:
                ener_geo[(n1, n2)] += flow
    return {e: f for e, f in ener_geo.items() if load_geo.get(e, 0) == 0}


def parse_od_nodes(filepath):
    """Return {od_key: (origin_node, dest_node)} from load flows."""
    solution = _parse_solution(filepath)
    T_max = max(edge[1][2]
                for rl in solution.values() for edge in rl[0])
    result = {}
    for od_key, route_list in solution.items():
        out_t0, in_Tmax = defaultdict(float), defaultdict(float)
        for edge, flow in route_list[0].items():
            if edge[0][2] == 0 and edge[0][0] in node_coords:
                out_t0[edge[0][0]] += flow
            if edge[1][2] == T_max and edge[1][0] in node_coords:
                in_Tmax[edge[1][0]] += flow
        result[od_key] = (
            max(out_t0,  key=out_t0.get)  if out_t0  else None,
            max(in_Tmax, key=in_Tmax.get) if in_Tmax else None,
        )
    return result


def parse_swap_events(filepath, n_dict):
    """
    Battery increases, edge duration ≤ 1 time step, node in n_dict.
    Returns {(od_key, node, t_start): flow}.
    """
    solution = _parse_solution(filepath)
    events = defaultdict(float)
    for od_key, route_list in solution.items():
        for edge, flow in route_list[0].items():
            n1, n2 = edge[0][0], edge[1][0]
            bat1, bat2 = edge[0][1], edge[1][1]
            t1, t2 = edge[0][2], edge[1][2]
            if (n1 == n2 and bat2 > bat1 and (t2 - t1) <= 1
                    and n1 in n_dict and n1 in node_coords and flow > 0):
                events[(od_key, n1, t1)] += flow
    return dict(events)


def parse_charge_sessions(filepath, n_dict, a_dict=None, station_proximity=1.0):
    """
    Battery increases at charger nodes (not in n_dict).
    If a_dict is provided and non-empty, restricts to nodes in a_dict;
    otherwise includes non-n nodes within station_proximity degrees of any
    existing charging station.
    Returns {(od_key, node): (t_start_hrs, t_end_hrs, flow)}.
    """
    solution = _parse_solution(filepath)
    raw = defaultdict(lambda: [float('inf'), -float('inf'), 0.0])
    for od_key, route_list in solution.items():
        for edge, flow in route_list[0].items():
            n1, n2 = edge[0][0], edge[1][0]
            bat1, bat2 = edge[0][1], edge[1][1]
            t1, t2 = edge[0][2], edge[1][2]
            if a_dict:
                in_scope = n1 in a_dict
            else:
                # keep only nodes near an existing charging station
                lat_n, lon_n = node_coords[n1]
                in_scope = any(
                    abs(node_coords[s][0] - lat_n) ** 2 +
                    abs(node_coords[s][1] - lon_n) ** 2 < station_proximity ** 2
                    for s in EXISTING_STATION_NODES if s in node_coords
                )
            if (n1 == n2 and bat2 > bat1 and n1 not in n_dict
                    and in_scope and n1 in node_coords and flow > 0):
                key = (od_key, n1)
                raw[key][0] = min(raw[key][0], t1)
                raw[key][1] = max(raw[key][1], t2)
                raw[key][2] = max(raw[key][2], flow)
    return {k: (v[0] * 15 / 60, v[1] * 15 / 60, v[2])
            for k, v in raw.items()}


def draw_load_flow(ax, load_flows):
    vals = list(load_flows.values())
    f_max = max(vals) if vals else 1.0
    for (n1, n2), flow in sorted(load_flows.items(), key=lambda x: x[1]):
        lat1, lon1 = node_coords[n1]
        lat2, lon2 = node_coords[n2]
        ax.annotate(
            '', xy=(lon2, lat2), xytext=(lon1, lat1),
            xycoords=PROJ._as_mpl_transform(ax),
            textcoords=PROJ._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle='->', color=LOAD_COLOR, alpha=0.75,
                lw=0.5 + 3.5 * (flow / f_max),
                mutation_scale=7 + 4 * (flow / f_max)),
            zorder=4)


def draw_repositioning_ellipses(ax, filepath, show_text=True):
    """
    Semi-transparent red ellipses showing where/when tractors reposition
    without a trailer. Clusters nearby nodes geographically, then splits
    each cluster into distinct time windows.
    """
    solution = _parse_solution(filepath)
    load_geo = defaultdict(float)
    for rl in solution.values():
        for e, v in rl[0].items():
            n1, n2 = e[0][0], e[1][0]
            if n1 != n2 and n1 in node_coords and n2 in node_coords:
                load_geo[(n1, n2)] += v

    # Collect (node, t_start) from energy-only edges
    ev_list = []
    for rl in solution.values():
        for edge, flow in rl[1].items():
            n1, n2 = edge[0][0], edge[1][0]
            if (n1 != n2 and n1 in node_coords and n2 in node_coords
                    and load_geo.get((n1, n2), 0) == 0 and flow > 0):
                t = edge[0][2]
                ev_list.append((n1, t))
                ev_list.append((n2, t))

    if not ev_list:
        return []

    placed_labels = []  # collect (lon, lat) of each repositioning label

    # Geographic clustering via union-find
    all_nodes = list({n for n, _ in ev_list})
    parent = {n: n for n in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    GEO_THRESH = 2.5  # degrees — nodes closer than this join one cluster
    for i, n1 in enumerate(all_nodes):
        lat1, lon1 = node_coords[n1]
        for n2 in all_nodes[i + 1:]:
            lat2, lon2 = node_coords[n2]
            if abs(lat1 - lat2) < GEO_THRESH and abs(lon1 - lon2) < GEO_THRESH:
                parent[find(n1)] = find(n2)

    geo_clusters = defaultdict(list)
    for n, t in ev_list:
        geo_clusters[find(n)].append((n, t))

    PAD      = 0.5   # degrees of padding around the bounding box
    MIN_AX   = 0.55  # minimum semi-axis length (degrees)
    TIME_GAP = 8     # time steps (2 h) — gap that separates distinct episodes

    for cluster_events in geo_clusters.values():
        # Ellipse geometry from all nodes in the cluster
        unique_nodes = list({n for n, _ in cluster_events})
        lats = [node_coords[n][0] for n in unique_nodes]
        lons = [node_coords[n][1] for n in unique_nodes]
        c_lat = (min(lats) + max(lats)) / 2
        c_lon = (min(lons) + max(lons)) / 2
        h = max((max(lats) - min(lats)) / 2 + PAD, MIN_AX)
        w = max((max(lons) - min(lons)) / 2 + PAD, MIN_AX)

        # Find distinct episode start times within this cluster
        sorted_ev = sorted(cluster_events, key=lambda x: x[1])
        episodes, ep = [], [sorted_ev[0]]
        for ev in sorted_ev[1:]:
            if ev[1] - ep[-1][1] > TIME_GAP:
                episodes.append(ep)
                ep = [ev]
            else:
                ep.append(ev)
        episodes.append(ep)
        start_hrs = sorted({ep[0][1] * 15 / 60 for ep in episodes})
        times_str = ', '.join(f'{h_:.3g}h' for h_ in start_hrs)

        ell = mpatches.Ellipse(
            (c_lon, c_lat), width=2 * w, height=2 * h,
            transform=PROJ, zorder=2,
            facecolor='#FF4444', alpha=0.18,
            edgecolor='#C62828', linewidth=1.2, linestyle='--')
        ax.add_patch(ell)

        # Place label above ellipse; flip below if it would cover a city label
        top_lat = c_lat + h + 0.1
        near_city = any(
            abs(node_coords[n][0] + 0.6 - top_lat) < 0.9 and
            abs(node_coords[n][1] - c_lon) < 1.5
            for n in CITY_LABELS if n in node_coords
        )
        lbl_lat = (c_lat - h - 0.7) if near_city else top_lat
        lbl_va  = 'top' if near_city else 'bottom'

        if show_text:
            ax.text(c_lon, lbl_lat,
                    f'Repositioning @ {times_str}',
                    fontsize=6, color='#C62828', fontweight='bold',
                    transform=PROJ, zorder=9, ha='center', va=lbl_va,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              alpha=0.85, edgecolor='#C62828', linewidth=0.6))
            placed_labels.append((c_lon, lbl_lat))

    return placed_labels


def _event_label(ax, lon, lat, text, color):
    ax.text(lon + 0.3, lat + 0.25, text,
            fontsize=6, color=color, fontweight='bold',
            transform=PROJ, zorder=8,
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      alpha=0.70, edgecolor='none'))


def draw_event_labels(ax, swap_events, charge_sessions, pre_placed=None):
    """
    Draw all swap (blue) and charge (green) text labels with greedy vertical
    stacking to prevent overlaps. pre_placed is a list of (lon, lat) positions
    already occupied by other labels (e.g. repositioning ellipse labels).
    """
    raw = []
    for (od_key, node, t_start), flow in swap_events.items():
        lat, lon = node_coords[node]
        raw.append((lon + 0.3, lat + 0.25, f'swap @ {t_start * 15 / 60:.2g}h', '#1565C0'))
    for (od_key, node), (t0_h, t1_h, flow) in charge_sessions.items():
        lat, lon = node_coords[node]
        raw.append((lon + 0.3, lat + 0.25, f'Charge @ {t0_h:.2g}h', '#2E7D32'))

    LON_TOL  = 1.8
    LAT_STEP = 0.30
    placed = list(pre_placed) if pre_placed else []
    for lon, lat, text, color in sorted(raw, key=lambda x: (round(x[0]), x[1])):
        adj = lat
        for _ in range(15):
            if not any(abs(p[0] - lon) < LON_TOL and abs(p[1] - adj) < LAT_STEP
                       for p in placed):
                break
            adj += LAT_STEP
        placed.append((lon, adj))
        ax.text(lon, adj, text,
                fontsize=6, color=color, fontweight='bold',
                transform=PROJ, zorder=8,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          alpha=0.70, edgecolor='none'))


def draw_city_labels(ax, od_nodes):
    """Orange dot + city abbreviation exactly at each origin/destination node."""
    labeled = set()
    for od_key, (origin, dest) in sorted(od_nodes.items()):
        for node in [origin, dest]:
            if node is not None and node not in labeled and node in CITY_LABELS:
                lat, lon = node_coords[node]
                ax.scatter(lon, lat, s=180, c='#FF8C00', marker='o', zorder=7,
                           transform=PROJ, edgecolors='#333333', linewidths=0.8)
                ax.text(lon, lat + 0.6, CITY_LABELS[node],
                        fontsize=8, fontweight='bold', color='#111111',
                        transform=PROJ, zorder=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.18', facecolor='#FF8C00',
                                  alpha=0.9, edgecolor='#333333', linewidth=0.8))
                labeled.add(node)


def draw_existing_stations(ax):
    """Dark grey squares for pre-existing real-world charging stations."""
    for nid in EXISTING_STATION_NODES:
        if nid in node_coords:
            lat, lon = node_coords[nid]
            ax.scatter(lon, lat, s=60, c='#555555', marker='s', zorder=4,
                       transform=PROJ, edgecolors='#222222', linewidths=0.5,
                       alpha=0.85)


# Charger stations from the charger_MMM result (used in Q5, Q6, Q7)
_a_q6 = records.get('charger_MMM', {}).get('a', {})
_n_q6 = {}

# ── Q5 — Flow map + swap events + repositioning (tractor_MMM) ─────────────────
load_flows_t    = parse_load_flows(records['tractor_MMM']['filepath'])
od_nodes_t      = parse_od_nodes(records['tractor_MMM']['filepath'])
swap_events     = parse_swap_events(records['tractor_MMM']['filepath'],
                                    records['tractor_MMM']['n'])
charge_sessions_t = parse_charge_sessions(records['tractor_MMM']['filepath'],
                                          records['tractor_MMM']['n'])

fig5, ax5 = plt.subplots(figsize=(14, 7), subplot_kw=dict(projection=PROJ))
setup_map(ax5)
repos_label_pos = draw_repositioning_ellipses(ax5, records['tractor_MMM']['filepath'])
draw_load_flow(ax5, load_flows_t)

for nid, count in records['tractor_MMM']['n'].items():
    lat, lon = node_coords[nid]
    ax5.scatter(lon, lat, s=count * 35, c='#1565C0', marker='o',
                alpha=0.9, zorder=5, transform=PROJ,
                edgecolors='white', linewidths=0.8)

for nid in _a_q6:
    if nid in node_coords:
        lat, lon = node_coords[nid]
        ax5.scatter(lon, lat, s=80, c='#B71C1C', marker='s', zorder=5,
                    transform=PROJ, edgecolors='white', linewidths=0.8, alpha=0.9)

draw_existing_stations(ax5)
draw_event_labels(ax5, swap_events, charge_sessions_t, pre_placed=repos_label_pos)
draw_city_labels(ax5, od_nodes_t)

flow_h    = mlines.Line2D([], [], color=LOAD_COLOR, linewidth=2.5, label='Truck flow (thickness ∝ amount)')
repos_h   = mpatches.Patch(facecolor='#FF4444', alpha=0.4, edgecolor='#C62828',
                            linestyle='--', label='Tractor repositioning / recharging')
station_h = mlines.Line2D([], [], color='#1565C0', marker='o', linestyle='None', markersize=8, label='Tractor swap station (size ∝ n)')
charger_h = mlines.Line2D([], [], color='#B71C1C', marker='s', linestyle='None', markersize=7, label='Charging-station-only solution')
exist_h   = mlines.Line2D([], [], color='#555555', marker='s', linestyle='None', markersize=7, label='Existing charging station')
ax5.legend(handles=[flow_h, repos_h, station_h, charger_h, exist_h],
           loc='lower left', fontsize=8, framealpha=0.9)
save(fig5, 'may25/q5_flow_tractor_mmm.png')

# ── Q6 — Flow map + charge events (charger_MMM, updated solution) ─────────────
_q6_sol_str = r"""{0: [{((56, np.int64(100), np.int64(0), 3), (20, np.int64(40), np.int64(11), 2)): 1.0, ((20, np.int64(40), np.int64(11), 2), (20, np.int64(80), np.int64(16), 3)): 1.0, ((20, np.int64(80), np.int64(16), 3), (20, np.int64(100), np.int64(17), 3)): 1.0, ((20, np.int64(100), np.int64(17), 3), (60, np.int64(30), np.int64(31), 1)): 1.0, ((60, np.int64(30), np.int64(31), 1), (60, np.int64(40), np.int64(33), 2)): 1.0, ((60, np.int64(40), np.int64(33), 2), (60, np.int64(80), np.int64(38), 3)): 1.0, ((60, np.int64(80), np.int64(38), 3), (60, np.int64(100), np.int64(39), 3)): 1.0, ((60, np.int64(100), np.int64(39), 3), (2, np.int64(40), np.int64(52), 2)): 1.0, ((2, np.int64(40), np.int64(52), 2), (2, np.int64(80), np.int64(57), 3)): 1.0, ((2, np.int64(80), np.int64(57), 3), (2, np.int64(100), np.int64(58), 3)): 1.0, ((2, np.int64(100), np.int64(58), 3), (69, np.int64(0), np.int64(78), 0)): 1.0, ((69, np.int64(0), np.int64(78), 0), (69, np.int64(0), 199, 0)): 1.0}, {((56, np.int64(100), np.int64(0), 3), (20, np.int64(40), np.int64(11), 2)): 1.0, ((20, np.int64(40), np.int64(11), 2), (20, np.int64(50), np.int64(13), 2)): 1.0, ((20, np.int64(50), np.int64(13), 2), (20, np.int64(80), np.int64(17), 3)): 1.0, ((20, np.int64(80), np.int64(17), 3), (20, np.int64(100), np.int64(18), 3)): 1.0, ((20, np.int64(100), np.int64(18), 3), (60, np.int64(30), np.int64(32), 1)): 1.0, ((60, np.int64(30), np.int64(32), 1), (60, np.int64(40), np.int64(34), 2)): 1.0, ((60, np.int64(40), np.int64(34), 2), (60, np.int64(60), np.int64(37), 2)): 1.0, ((60, np.int64(60), np.int64(37), 2), (60, np.int64(80), np.int64(40), 3)): 1.0, ((60, np.int64(80), np.int64(40), 3), (60, np.int64(90), np.int64(41), 3)): 1.0, ((60, np.int64(90), np.int64(41), 3), (60, np.int64(100), np.int64(42), 3)): 1.0, ((60, np.int64(100), np.int64(42), 3), (2, np.int64(40), np.int64(55), 2)): 1.0, ((2, np.int64(40), np.int64(55), 2), (62, np.int64(40), np.int64(58), 2)): 1.0, ((62, np.int64(40), np.int64(58), 2), (101, np.int64(40), np.int64(59), 2)): 1.0, ((101, np.int64(40), np.int64(59), 2), (22, np.int64(40), np.int64(61), 2)): 1.0, ((22, np.int64(40), np.int64(61), 2), (22, np.int64(80), np.int64(66), 3)): 1.0, ((22, np.int64(80), np.int64(66), 3), (22, np.int64(100), np.int64(67), 3)): 1.0, ((22, np.int64(100), np.int64(67), 3), (101, np.int64(100), np.int64(69), 3)): 1.0, ((101, np.int64(100), np.int64(69), 3), (62, np.int64(100), np.int64(70), 3)): 1.0, ((62, np.int64(100), np.int64(70), 3), (2, np.int64(100), np.int64(73), 3)): 1.0, ((2, np.int64(100), np.int64(73), 3), (69, np.int64(0), np.int64(93), 0)): 1.0, ((69, np.int64(0), np.int64(93), 0), (69, np.int64(0), 199, 0)): 1.0}], 1: [{((90, np.int64(80), np.int64(0), 3), (90, np.int64(100), np.int64(1), 3)): 1.0, ((90, np.int64(100), np.int64(1), 3), (0, np.int64(30), np.int64(14), 1)): 1.0, ((0, np.int64(30), np.int64(14), 1), (0, np.int64(40), np.int64(16), 2)): 1.0, ((0, np.int64(40), np.int64(16), 2), (0, np.int64(80), np.int64(21), 3)): 1.0, ((0, np.int64(80), np.int64(21), 3), (0, np.int64(100), np.int64(22), 3)): 1.0, ((0, np.int64(100), np.int64(22), 3), (89, np.int64(10), np.int64(36), 1)): 1.0, ((89, np.int64(10), np.int64(36), 1), (89, np.int64(40), np.int64(41), 2)): 1.0, ((89, np.int64(40), np.int64(41), 2), (89, np.int64(50), np.int64(43), 2)): 1.0, ((89, np.int64(50), np.int64(43), 2), (21, np.int64(50), np.int64(44), 2)): 1.0, ((21, np.int64(50), np.int64(44), 2), (89, np.int64(50), np.int64(45), 2)): 1.0, ((89, np.int64(50), np.int64(45), 2), (89, np.int64(70), np.int64(46), 2)): 1.0, ((89, np.int64(70), np.int64(46), 2), (89, np.int64(70), 199, 2)): 1.0}, {((90, np.int64(100), np.int64(0), 3), (71, np.int64(100), np.int64(1), 3)): 1.0, ((71, np.int64(100), np.int64(1), 3), (28, np.int64(100), np.int64(2), 3)): 1.0, ((28, np.int64(100), np.int64(2), 3), (90, np.int64(100), np.int64(4), 3)): 1.0, ((90, np.int64(100), np.int64(4), 3), (0, np.int64(30), np.int64(17), 1)): 1.0, ((0, np.int64(30), np.int64(17), 1), (0, np.int64(40), np.int64(19), 2)): 1.0, ((0, np.int64(40), np.int64(19), 2), (0, np.int64(80), np.int64(24), 3)): 1.0, ((0, np.int64(80), np.int64(24), 3), (0, np.int64(100), np.int64(25), 3)): 1.0, ((0, np.int64(100), np.int64(25), 3), (89, np.int64(10), np.int64(39), 1)): 1.0, ((89, np.int64(10), np.int64(39), 1), (89, np.int64(20), np.int64(41), 1)): 1.0, ((89, np.int64(20), np.int64(41), 1), (89, np.int64(40), np.int64(45), 2)): 1.0, ((89, np.int64(40), np.int64(45), 2), (89, np.int64(50), np.int64(47), 2)): 1.0, ((89, np.int64(50), np.int64(47), 2), (21, np.int64(50), np.int64(48), 2)): 1.0, ((21, np.int64(50), np.int64(48), 2), (89, np.int64(50), np.int64(49), 2)): 1.0, ((89, np.int64(50), np.int64(49), 2), (89, np.int64(70), np.int64(50), 2)): 1.0, ((89, np.int64(70), np.int64(50), 2), (89, np.int64(70), 199, 2)): 1.0}], 2: [{((56, np.int64(90), np.int64(0), 3), (56, np.int64(100), np.int64(1), 3)): 5.0, ((56, np.int64(100), np.int64(1), 3), (64, np.int64(80), np.int64(10), 3)): 5.0, ((64, np.int64(80), np.int64(10), 3), (5, np.int64(0), np.int64(27), 0)): 5.0, ((5, np.int64(0), np.int64(27), 0), (5, np.int64(10), np.int64(29), 1)): 5.0, ((5, np.int64(10), np.int64(29), 1), (5, np.int64(40), np.int64(34), 2)): 5.0, ((5, np.int64(40), np.int64(34), 2), (5, np.int64(80), np.int64(39), 3)): 5.0, ((5, np.int64(80), np.int64(39), 3), (5, np.int64(100), np.int64(40), 3)): 5.0, ((5, np.int64(100), np.int64(40), 3), (17, np.int64(10), np.int64(59), 1)): 5.0, ((17, np.int64(10), np.int64(59), 1), (17, np.int64(40), np.int64(64), 2)): 5.0, ((17, np.int64(40), np.int64(64), 2), (17, np.int64(80), np.int64(69), 3)): 5.0, ((17, np.int64(80), np.int64(69), 3), (17, np.int64(100), np.int64(70), 3)): 5.0, ((17, np.int64(100), np.int64(70), 3), (24, np.int64(90), np.int64(74), 3)): 5.0, ((24, np.int64(90), np.int64(74), 3), (89, np.int64(0), np.int64(92), 0)): 5.0, ((89, np.int64(0), np.int64(92), 0), (89, np.int64(10), np.int64(94), 1)): 5.0, ((89, np.int64(10), np.int64(94), 1), (89, np.int64(40), np.int64(99), 2)): 5.0, ((89, np.int64(40), np.int64(99), 2), (89, np.int64(50), np.int64(101), 2)): 5.0, ((89, np.int64(50), np.int64(101), 2), (21, np.int64(50), np.int64(102), 2)): 5.0, ((21, np.int64(50), np.int64(102), 2), (21, np.int64(70), np.int64(103), 2)): 5.0, ((21, np.int64(70), np.int64(103), 2), (21, np.int64(70), 199, 2)): 5.0, ((91, np.int64(100), np.int64(0), 3), (60, np.int64(90), np.int64(4), 3)): 5.0, ((60, np.int64(90), np.int64(4), 3), (60, np.int64(100), np.int64(5), 3)): 5.0, ((60, np.int64(100), np.int64(5), 3), (20, np.int64(30), np.int64(19), 1)): 5.0, ((20, np.int64(30), np.int64(19), 1), (20, np.int64(40), np.int64(21), 2)): 5.0, ((20, np.int64(40), np.int64(21), 2), (20, np.int64(60), np.int64(24), 2)): 5.0, ((20, np.int64(60), np.int64(24), 2), (56, np.int64(0), np.int64(35), 0)): 5.0, ((56, np.int64(0), np.int64(35), 0), (56, np.int64(0), 199, 0)): 5.0}, {((91, np.int64(100), np.int64(0), 3), (60, np.int64(90), np.int64(4), 3)): 5.0, ((60, np.int64(90), np.int64(4), 3), (60, np.int64(100), np.int64(5), 3)): 5.0, ((60, np.int64(100), np.int64(5), 3), (20, np.int64(30), np.int64(19), 1)): 5.0, ((20, np.int64(30), np.int64(19), 1), (20, np.int64(40), np.int64(21), 2)): 5.0, ((20, np.int64(40), np.int64(21), 2), (20, np.int64(60), np.int64(24), 2)): 5.0, ((20, np.int64(60), np.int64(24), 2), (56, np.int64(0), np.int64(35), 0)): 5.0, ((56, np.int64(0), np.int64(35), 0), (11, np.int64(0), np.int64(38), 0)): 5.0, ((11, np.int64(0), np.int64(38), 0), (11, np.int64(10), np.int64(40), 1)): 5.0, ((11, np.int64(10), np.int64(40), 1), (11, np.int64(30), np.int64(44), 1)): 5.0, ((11, np.int64(30), np.int64(44), 1), (11, np.int64(40), np.int64(46), 2)): 5.0, ((11, np.int64(40), np.int64(46), 2), (11, np.int64(60), np.int64(49), 2)): 5.0, ((11, np.int64(60), np.int64(49), 2), (11, np.int64(80), np.int64(52), 3)): 5.0, ((11, np.int64(80), np.int64(52), 3), (11, np.int64(100), np.int64(53), 3)): 5.0, ((11, np.int64(100), np.int64(53), 3), (56, np.int64(100), np.int64(56), 3)): 5.0, ((56, np.int64(100), np.int64(56), 3), (64, np.int64(80), np.int64(65), 3)): 5.0, ((64, np.int64(80), np.int64(65), 3), (5, np.int64(0), np.int64(82), 0)): 5.0, ((5, np.int64(0), np.int64(82), 0), (5, np.int64(10), np.int64(84), 1)): 5.0, ((5, np.int64(10), np.int64(84), 1), (5, np.int64(40), np.int64(89), 2)): 5.0, ((5, np.int64(40), np.int64(89), 2), (5, np.int64(80), np.int64(94), 3)): 5.0, ((5, np.int64(80), np.int64(94), 3), (5, np.int64(100), np.int64(95), 3)): 5.0, ((5, np.int64(100), np.int64(95), 3), (17, np.int64(10), np.int64(114), 1)): 5.0, ((17, np.int64(10), np.int64(114), 1), (7, np.int64(10), np.int64(115), 1)): 5.0, ((7, np.int64(10), np.int64(115), 1), (7, np.int64(20), np.int64(117), 1)): 5.0, ((7, np.int64(20), np.int64(117), 1), (17, np.int64(20), np.int64(118), 1)): 5.0, ((17, np.int64(20), np.int64(118), 1), (7, np.int64(20), np.int64(119), 1)): 5.0, ((7, np.int64(20), np.int64(119), 1), (7, np.int64(30), np.int64(121), 1)): 5.0, ((7, np.int64(30), np.int64(121), 1), (7, np.int64(40), np.int64(123), 2)): 5.0, ((7, np.int64(40), np.int64(123), 2), (17, np.int64(40), np.int64(124), 2)): 5.0, ((17, np.int64(40), np.int64(124), 2), (7, np.int64(40), np.int64(125), 2)): 5.0, ((7, np.int64(40), np.int64(125), 2), (7, np.int64(50), np.int64(127), 2)): 5.0, ((7, np.int64(50), np.int64(127), 2), (7, np.int64(70), np.int64(128), 2)): 5.0, ((7, np.int64(70), np.int64(128), 2), (17, np.int64(70), np.int64(129), 2)): 5.0, ((17, np.int64(70), np.int64(129), 2), (24, np.int64(60), np.int64(133), 2)): 5.0, ((24, np.int64(60), np.int64(133), 2), (24, np.int64(60), 199, 2)): 5.0}]}"""
_SOLUTION_OVERRIDES[records['charger_MMM']['filepath']] = ast.literal_eval(
    re.sub(r'np\.int64\((\d+)\)', r'\1', _q6_sol_str))

load_flows_c    = parse_load_flows(records['charger_MMM']['filepath'])
od_nodes_c      = parse_od_nodes(records['charger_MMM']['filepath'])
charge_sessions = parse_charge_sessions(records['charger_MMM']['filepath'],
                                        _n_q6, _a_q6)

fig6, ax6 = plt.subplots(figsize=(14, 7), subplot_kw=dict(projection=PROJ))
setup_map(ax6)
draw_load_flow(ax6, load_flows_c)

for nid, count in _a_q6.items():
    lat, lon = node_coords[nid]
    ax6.scatter(lon, lat, s=max(count, 1) * 120, c='#B71C1C', marker='s',
                alpha=0.9, zorder=5, transform=PROJ,
                edgecolors='white', linewidths=0.8)
    label_node(ax6, lon, lat, f'{nid}\n(a={int(count)})',
               color='#B71C1C', fontsize=6)

draw_existing_stations(ax6)
draw_event_labels(ax6, {}, charge_sessions)
draw_city_labels(ax6, od_nodes_c)

# Dashed extensions for incomplete corridors
# OD1: AMA (89) → PHX (56),  OD2: ELK (43) → DAL (90)
for n_from, n_to in [(89, 56), (43, 90)]:
    lat1, lon1 = node_coords[n_from]
    lat2, lon2 = node_coords[n_to]
    ax6.annotate(
        '', xy=(lon2, lat2), xytext=(lon1, lat1),
        xycoords=PROJ._as_mpl_transform(ax6),
        textcoords=PROJ._as_mpl_transform(ax6),
        arrowprops=dict(arrowstyle='->', color='#888888',
                        lw=1.5, linestyle='dashed',
                        mutation_scale=10),
        zorder=3)

flow_h_c    = mlines.Line2D([], [], color=LOAD_COLOR, linewidth=2.5, label='Truck flow (thickness ∝ amount)')
ext_h       = mlines.Line2D([], [], color='#888888', linewidth=1.5, linestyle='--', label='Planned extension (time exceeded)')
station_h_c = mlines.Line2D([], [], color='#B71C1C', marker='s', linestyle='None', markersize=8, label='Charging-station-only solution')
exist_h_c   = mlines.Line2D([], [], color='#555555', marker='s', linestyle='None', markersize=7, label='Existing charging station')
ax6.legend(handles=[flow_h_c, ext_h, station_h_c, exist_h_c],
           loc='lower left', fontsize=8, framealpha=0.9)

save(fig6, 'may25/q6_flow_charger_mmm.png')

# ── Q7 — Flow map + swap events (battery_MMM) ─────────────────────────────────
load_flows_b    = parse_load_flows(records['battery_MMM']['filepath'])
od_nodes_b      = parse_od_nodes(records['battery_MMM']['filepath'])
swap_events_b   = parse_swap_events(records['battery_MMM']['filepath'],
                                    records['battery_MMM']['n'])
charge_sessions_b = parse_charge_sessions(records['battery_MMM']['filepath'],
                                          records['battery_MMM']['n'])

fig7, ax7 = plt.subplots(figsize=(14, 7), subplot_kw=dict(projection=PROJ))
setup_map(ax7)
draw_load_flow(ax7, load_flows_b)

for nid, count in records['battery_MMM']['n'].items():
    lat, lon = node_coords[nid]
    ax7.scatter(lon, lat, s=count * 35, c='#1565C0', marker='o',
                alpha=0.9, zorder=5, transform=PROJ,
                edgecolors='white', linewidths=0.8)

for nid in _a_q6:
    if nid in node_coords:
        lat, lon = node_coords[nid]
        ax7.scatter(lon, lat, s=80, c='#B71C1C', marker='s', zorder=5,
                    transform=PROJ, edgecolors='white', linewidths=0.8, alpha=0.9)

draw_existing_stations(ax7)
charge_sessions_b_filtered = {k: v for k, v in charge_sessions_b.items() if v[0] > 0}
draw_event_labels(ax7, swap_events_b, charge_sessions_b_filtered)
draw_city_labels(ax7, od_nodes_b)

flow_h7     = mlines.Line2D([], [], color=LOAD_COLOR, linewidth=2.5, label='Truck flow (thickness ∝ amount)')
station_h7  = mlines.Line2D([], [], color='#1565C0', marker='o', linestyle='None', markersize=8, label='Battery swap station (size ∝ n)')
charger_h7  = mlines.Line2D([], [], color='#B71C1C', marker='s', linestyle='None', markersize=7, label='Charging-station-only solution')
exist_h7    = mlines.Line2D([], [], color='#555555', marker='s', linestyle='None', markersize=7, label='Existing charging station')
ax7.legend(handles=[flow_h7, station_h7, charger_h7, exist_h7],
           loc='lower left', fontsize=8, framealpha=0.9)
save(fig7, 'may25/q7_flow_battery_mmm.png')

# ── Q8 — Network: existing stations and proxy sites ───────────────────────────
fig8, ax8 = plt.subplots(figsize=(14, 7), subplot_kw=dict(projection=PROJ))
setup_map(ax8)

# Proxy sites: all network nodes that are NOT existing stations
for nid, (lat, lon) in node_coords.items():
    if nid not in EXISTING_STATION_NODES:
        ax8.scatter(lon, lat, s=30, c='#1565C0', marker='o', zorder=3,
                    transform=PROJ, edgecolors='#0D47A1', linewidths=0.4, alpha=0.8)

# Existing stations on top
for nid in EXISTING_STATION_NODES:
    if nid in node_coords:
        lat, lon = node_coords[nid]
        ax8.scatter(lon, lat, s=80, c='#333333', marker='s', zorder=4,
                    transform=PROJ, edgecolors='#111111', linewidths=0.5, alpha=0.9)

proxy_h  = mlines.Line2D([], [], color='#1565C0', marker='o', linestyle='None',
                          markersize=7, label='Proxy site')
exist_h8 = mlines.Line2D([], [], color='#333333', marker='s', linestyle='None',
                          markersize=7, label='Existing charging station')
ax8.legend(handles=[proxy_h, exist_h8], loc='lower left', fontsize=9, framealpha=0.9)
save(fig8, 'may25/q8_network_sites.png')
