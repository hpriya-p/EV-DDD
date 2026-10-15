import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from collections import defaultdict

# ============================================================
# Solution data
# ============================================================
x_load = {((10, 69, 6, 2), (23, 6, 11, 0), 0): 24.0, ((3, 72, 1, 2), (3, 100, 1, 3), 0): 24.0, ((34, 45, 11, 2), (34, 80, 11, 3), 0): 14.0, ((3, 100, 1, 3), (10, 43, 6, 2), 0): 24.0, ((25, 19, 6, 1), (25, 100, 6, 3), 0): 14.0, ((10, 43, 6, 2), (10, 69, 6, 2), 0): 24.0, ((23, 6, 11, 0), (23, 6, 35, 0), 0): 24.0, ((18, 0, 21, 0), (18, 0, 35, 0), 0): 14.0, ((34, 80, 11, 3), (18, 0, 21, 0), 0): 14.0, ((25, 100, 6, 3), (34, 45, 11, 2), 0): 14.0, ((29, 76, 0, 2), (25, 19, 6, 1), 0): 14.0, ((29, 100, 0, 3), (3, 72, 1, 2), 0): 24.0}
x_ener = {((0, 62, 1, 2), (9, 62, 1, 2), 0): 8568.0, ((9, 59, 1, 2), (2, 59, 1, 2), 0): 8533.0, ((2, 59, 1, 2), (0, 62, 1, 2), 0): 4284.0, ((2, 59, 1, 2), (9, 59, 1, 2), 0): 8498.0, ((9, 62, 1, 2), (0, 62, 1, 2), 0): 4249.0, ((9, 62, 1, 2), (2, 59, 1, 2), 0): 4249.0, ((9, 62, 1, 2), (2, 62, 1, 2), 0): 70.0, ((2, 62, 1, 2), (0, 62, 1, 2), 0): 35.0, ((2, 62, 1, 2), (9, 59, 1, 2), 0): 35.0, ((10, 69, 6, 2), (23, 6, 11, 0), 0): 24.0, ((4, 73, 6, 2), (10, 69, 6, 2), 0): 24.0, ((8, 80, 6, 3), (4, 73, 6, 2), 0): 24.0, ((8, 52, 11, 2), (26, 17, 16, 1), 0): 24.0, ((26, 17, 16, 1), (26, 25, 16, 1), 0): 24.0, ((26, 25, 16, 1), (26, 25, 21, 1), 0): 24.0, ((26, 25, 21, 1), (26, 26, 21, 1), 0): 48.0, ((9, 17, 16, 1), (2, 14, 16, 1), 0): 34421726.0, ((2, 14, 16, 1), (9, 17, 16, 1), 0): 34421726.0, ((9, 17, 21, 1), (2, 14, 21, 1), 0): 7247484.0, ((2, 14, 21, 1), (9, 17, 21, 1), 0): 7247484.0, ((9, 17, 26, 1), (2, 14, 26, 1), 0): 1287707.0, ((2, 14, 26, 1), (9, 17, 26, 1), 0): 1287707.0, ((9, 17, 31, 1), (2, 14, 31, 1), 0): 125630.0, ((2, 14, 31, 1), (9, 17, 31, 1), 0): 125630.0, ((3, 72, 1, 2), (3, 72, 6, 2), 0): 24.0, ((3, 72, 6, 2), (8, 52, 11, 2), 0): 24.0, ((34, 45, 11, 2), (34, 45, 16, 2), 0): 14.0, ((34, 45, 16, 2), (19, 6, 21, 0), 0): 14.0, ((19, 6, 21, 0), (19, 6, 35, 0), 0): 14.0, ((3, 100, 1, 3), (8, 80, 6, 3), 0): 24.0, ((3, 100, 1, 3), (10, 43, 6, 2), 0): 24.0, ((25, 19, 6, 1), (25, 19, 35, 1), 0): 14.0, ((10, 43, 6, 2), (10, 43, 11, 2), 0): 24.0, ((10, 43, 11, 2), (10, 43, 16, 2), 0): 24.0, ((10, 43, 16, 2), (10, 43, 21, 2), 0): 24.0, ((10, 43, 21, 2), (26, 25, 21, 1), 0): 24.0, ((23, 6, 11, 0), (23, 6, 16, 0), 0): 24.0, ((23, 6, 16, 0), (23, 6, 35, 0), 0): 24.0, ((26, 31, 21, 1), (26, 31, 26, 1), 0): 48.0, ((26, 31, 26, 1), (23, 6, 31, 0), 0): 48.0, ((23, 6, 31, 0), (23, 6, 35, 0), 0): 48.0, ((26, 26, 21, 1), (26, 31, 21, 1), 0): 48.0, ((11, 98, 1, 3), (28, 100, 1, 3), 0): 48.0, ((28, 100, 1, 3), (3, 100, 1, 3), 0): 48.0, ((18, 0, 21, 0), (18, 0, 26, 0), 0): 14.0, ((18, 0, 26, 0), (18, 0, 31, 0), 0): 14.0, ((18, 0, 31, 0), (18, 0, 35, 0), 0): 14.0, ((19, 100, 1, 3), (34, 80, 6, 3), 0): 14.0, ((19, 100, 0, 3), (19, 100, 1, 3), 0): 14.0, ((34, 80, 6, 3), (34, 80, 11, 3), 0): 14.0, ((34, 80, 11, 3), (18, 0, 21, 0), 0): 14.0, ((25, 100, 1, 3), (27, 100, 1, 3), 0): 14.0, ((25, 100, 0, 3), (25, 100, 1, 3), 0): 14.0, ((27, 100, 1, 3), (27, 100, 6, 3), 0): 14.0, ((25, 100, 6, 3), (34, 45, 11, 2), 0): 14.0, ((27, 100, 6, 3), (25, 100, 6, 3), 0): 14.0, ((29, 76, 0, 2), (25, 19, 6, 1), 0): 14.0, ((29, 100, 0, 3), (3, 72, 1, 2), 0): 24.0, ((29, 100, 0, 3), (11, 98, 1, 3), 0): 48.0}
a = {26: 1.0}
n = {19: 14.0, 25: 14.0}
pos_latlon = {0: (34.421443, -119.684178), 1: (33.862772, -117.234706), 2: (34.4349097, -119.9212861), 3: (34.689589, -118.130371), 4: (35.400164, -119.399727), 5: (38.7747705, -119.9029911), 6: (33.9541734, -117.5556971), 7: (36.009923, -114.784826), 8: (35.3002628, -119.0381956), 9: (34.4425082, -119.7812613), 10: (36.075856, -119.043996), 11: (33.8722233, -118.2654222), 12: (35.263617, -112.194139), 13: (33.7885884, -118.2150827), 14: (32.596725, -117.019495), 15: (34.886448, -117.079423), 16: (35.9874334, -119.961999), 17: (35.01643, -110.675005), 18: (33.30006, -111.96835), 19: (32.794671, -115.539473), 20: (36.220567, -115.125553), 21: (33.3021433, -111.9552863), 22: (33.39639, -110.76888), 23: (37.8135094, -119.8800771), 24: (37.161682, -113.434692), 25: (34.137674, -116.303336), 26: (36.72136, -119.761165), 27: (33.8310496, -116.5452277), 28: (34.12755, -118.443664), 29: (33.899899, -118.207088), 30: (33.0390751, -117.2841996), 31: (33.3874378, -110.7609849), 32: (32.178328, -110.960207), 33: (38.164391, -112.278183), 34: (33.603063, -114.598023)}
# ============================================================
# Helpers
# ============================================================

def parse_val(s):
    s = str(s)
    if 'int64' in s:
        return int(s.split('(')[1].rstrip(')'))
    try:
        return int(s)
    except ValueError:
        return s


def aggregate_transit(flow_dict):
    """Sum flows by physical (i→j) edge, skipping same-node arcs."""
    flows = defaultdict(float)
    for (src, dst, _), val in flow_dict.items():
        i, j = src[0], dst[0]
        if i != j:
            flows[(i, j)] += val
    return dict(flows)


def find_charges(flow_dict):
    """Collect charging arcs (same node, battery level changes)."""
    events = defaultdict(list)
    for (src, dst, _), val in flow_dict.items():
        i, j = src[0], dst[0]
        if i == j:
            g1 = parse_val(src[1])
            g2 = parse_val(dst[1])
            if g1 != g2:
                events[i].append((g1, g2, val))
    return dict(events)


def draw_arrow(ax, lon1, lat1, lon2, lon2_lat, proj, color, lw, alpha=0.85, zorder=5):
    """Draw a line with an arrowhead at the destination."""
    ax.plot([lon1, lon2], [lat1, lon2_lat], '-', color=color, linewidth=lw,
            alpha=alpha, transform=proj, zorder=zorder)
    # Arrowhead: triangle marker rotated toward destination
    dlat = lon2_lat - lat1
    dlon = lon2 - lon1
    angle = math.degrees(math.atan2(dlat, dlon))
    ax.plot(lon2, lon2_lat, marker=(3, 0, -(angle - 90)), markersize=8 + lw,
            color=color, alpha=alpha, transform=proj, zorder=zorder + 1,
            markeredgecolor='none')


# ============================================================
# Build plot
# ============================================================

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)

lons = [lon for lat, lon in pos_latlon.values()]
lats = [lat for lat, lon in pos_latlon.values()]
margin = 0.8
ax.set_extent([min(lons) - margin, max(lons) + margin,
               min(lats) - margin, max(lats) + margin], crs=proj)

ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#cce5f0', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='gray', zorder=1)
ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.4, linestyle='--')

# --- All nodes (background) ---
for node_id, (lat, lon) in pos_latlon.items():
    ax.plot(lon, lat, 'o', color='#cccccc', markersize=5, transform=proj, zorder=2,
            markeredgecolor='#999999', markeredgewidth=0.5)
    ax.text(lon + 0.08, lat + 0.06, str(node_id), fontsize=5.5,
            transform=proj, zorder=3, color='#666666')

# --- x_ener transit flows (blue, drawn under x_load) ---
ener_transit = aggregate_transit(x_ener)
max_ener = max(ener_transit.values()) if ener_transit else 1

for (i, j), flow in sorted(ener_transit.items(), key=lambda x: x[1]):
    lat1, lon1 = pos_latlon[i]
    lat2, lon2 = pos_latlon[j]
    lw = 0.5 + 3.5 * math.log1p(flow) / math.log1p(max_ener)
    draw_arrow(ax, lon1, lat1, lon2, lat2, proj, color='steelblue', lw=lw, alpha=0.5, zorder=4)

# --- x_load transit flows (red, on top) ---
load_transit = aggregate_transit(x_load)
max_load = max(load_transit.values()) if load_transit else 1

for (i, j), flow in sorted(load_transit.items(), key=lambda x: x[1]):
    lat1, lon1 = pos_latlon[i]
    lat2, lon2 = pos_latlon[j]
    lw = 1.5 + 3.5 * (flow / max_load)
    draw_arrow(ax, lon1, lat1, lon2, lat2, proj, color='tomato', lw=lw, alpha=0.9, zorder=5)
    mid_lon = (lon1 + lon2) / 2
    mid_lat = (lat1 + lat2) / 2
    ax.text(mid_lon, mid_lat, str(int(flow)), fontsize=7.5, transform=proj,
            ha='center', va='center', color='darkred', fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

# --- Charging events (gold star) ---
for flow_dict, label in [(x_load, 'load'), (x_ener, 'ener')]:
    for node_id, events in find_charges(flow_dict).items():
        lat, lon = pos_latlon[node_id]
        ax.plot(lon, lat, '*', color='gold', markersize=20, markeredgecolor='black',
                markeredgewidth=1.5, transform=proj, zorder=8)
        g1 = min(e[0] for e in events)
        g2 = max(e[1] for e in events)
        ax.text(lon + 0.15, lat + 0.25, f'⚡{g1}→{g2}%', fontsize=7,
                transform=proj, color='goldenrod', fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

# --- Nodes with charger installed (a > 0, green square) ---
for node_id, val in a.items():
    if val > 0:
        lat, lon = pos_latlon[node_id]
        ax.plot(lon, lat, 's', color='limegreen', markersize=14, markeredgecolor='black',
                markeredgewidth=1.5, transform=proj, zorder=9)
        ax.text(lon, lat - 0.4, f'a[{node_id}]={int(val)}', fontsize=8,
                transform=proj, ha='center', color='darkgreen', fontweight='bold', zorder=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='darkgreen'))

# --- Nodes with tractors stationed (n > 0, orange triangle) ---
for node_id, val in n.items():
    if val > 0:
        lat, lon = pos_latlon[node_id]
        ax.plot(lon, lat, '^', color='orange', markersize=14, markeredgecolor='black',
                markeredgewidth=1.5, transform=proj, zorder=9)
        ax.text(lon, lat + 0.4, f'n[{node_id}]={int(val)}', fontsize=8,
                transform=proj, ha='center', color='darkorange', fontweight='bold', zorder=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='darkorange'))

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color='tomato', linewidth=3, label=f'x_load truck flow ({len(load_transit)} arcs)'),
    Line2D([0], [0], color='steelblue', linewidth=2, label=f'x_ener energy flow ({len(ener_transit)} arcs, log-scaled)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black',
           markersize=14, label='Charging event'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='limegreen', markeredgecolor='black',
           markersize=10, label='Charger installed (a=1)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markeredgecolor='black',
           markersize=10, label='Tractors stationed (n>0)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

n_tractors = sum(int(v) for v in n.values())
n_chargers = sum(int(v) for v in a.values())
ax.set_title(
    f'Tri-City Experiment: Flow Solution\n'
    f'Truck flow arcs: {len(load_transit)}  |  Energy flow arcs: {len(ener_transit)}  |  '
    f'Chargers installed: {n_chargers}  |  Tractors stationed: {n_tractors}',
    fontsize=12
)

plt.tight_layout()
plt.savefig('tri_city_solution_plot.png', dpi=150, bbox_inches='tight')
print("Saved to tri_city_solution_plot.png")
plt.show()
