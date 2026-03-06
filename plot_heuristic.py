import json
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Read network and build graph
df = pd.read_csv("data/SiouxFalls_net.tntp", sep='\t', lineterminator='\n')
N = nx.DiGraph()
for ind, row in df.iterrows():
    t = row['free_flow_time']
    dist = int(t * 0.01 * 100)
    N.add_edge(int(row['init_node']), int(row['term_node']), dH=dist, dL=int(dist/2), time=t)

# Standard Sioux Falls node positions
pos = {
    1: (0, 6), 2: (1, 6), 3: (0, 5), 4: (1, 5), 5: (2, 5), 6: (2, 6),
    7: (3, 7), 8: (3, 6), 9: (3, 5), 10: (3, 4), 11: (2, 4), 12: (1, 4),
    13: (0, 3), 14: (2, 3), 15: (3, 3), 16: (4, 5), 17: (4, 4), 18: (4, 6),
    19: (4, 3), 20: (5, 4), 21: (5, 2), 22: (4, 2), 23: (3, 2), 24: (2, 2)
}

# Load solution from JSON
with open("results/experiment_dee318c0-heuristic-tractor.json") as f:
    data = json.load(f)

soln = data['solution']

# Convert string tuple keys back to actual tuples
def parse_flow(flow_dict):
    parsed = {}
    for k, v in flow_dict.items():
        parsed[ast.literal_eval(k)] = v
    return parsed

x_load = parse_flow(soln['x_load'])
x_ener = parse_flow(soln['x_ener'])
a = {int(k): v for k, v in soln['a'].items()}
n = {int(k): v for k, v in soln['n'].items()}

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Draw base network (light gray)
nx.draw_networkx_edges(N, pos, edge_color='lightgray', alpha=0.5, ax=ax)
nx.draw_networkx_nodes(N, pos, node_color='lightblue', node_size=500, ax=ax)
nx.draw_networkx_labels(N, pos, font_size=10, ax=ax)

# Decompose flow into paths
def flow_decomposition(flow):
    """Decompose a flow into distinct paths from sources to sinks."""
    adj = {}
    in_degree = {}
    out_degree = {}

    for arc_key in flow.keys():
        src, dst = arc_key
        src = tuple(src)
        dst = tuple(dst)

        if src not in adj:
            adj[src] = []
        adj[src].append((dst, (src, dst)))

        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
        if src not in in_degree:
            in_degree[src] = 0
        if dst not in out_degree:
            out_degree[dst] = 0

    # Find source nodes (in_degree == 0)
    sources = [node for node in in_degree if in_degree[node] == 0]

    # Trace paths: keep extracting paths until all arcs are consumed
    paths = []
    while True:
        # Find a node that still has outgoing arcs, preferring sources
        start = None
        for s in sources:
            if s in adj and adj[s]:
                start = s
                break
        if start is None:
            # Check any node with remaining outgoing arcs
            for node in list(adj.keys()):
                if adj[node]:
                    start = node
                    break
        if start is None:
            break

        path = []
        current = start
        while current in adj and adj[current]:
            next_node, arc_key = adj[current].pop(0)
            path.append(arc_key)
            current = next_node
        if path:
            paths.append(path)

    return paths

# Global counter for label positions on each physical edge (across all plot_path calls)
global_edge_label_idx = {}

# Plot a single path (adapted for 4-tuples: node, battery, time, index)
def plot_path(path, path_color, rad):
    path_edges = [(src[0], dst[0]) for src, dst in path if src[0] != dst[0]]
    path_nodes = set()
    for src, dst in path:
        path_nodes.add(src[0])
        path_nodes.add(dst[0])

    # Draw edges
    nx.draw_networkx_edges(N, pos, edgelist=path_edges, edge_color=path_color,
                           width=3, arrows=True, arrowsize=20, ax=ax,
                           connectionstyle=f"arc3,rad={rad}")

    # Add annotations for battery levels and time
    for src, dst in path:
        i, g1, t1, _ = src
        j, g2, t2, _ = dst
        if i != j:
            key = (i, j)
            global_edge_label_idx[key] = global_edge_label_idx.get(key, 0) + 1
            occurrence = global_edge_label_idx[key]

            x1, y1 = pos[i]
            x2, y2 = pos[j]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                along_x = dx / length
                along_y = dy / length
            else:
                perp_x, perp_y = 0, 0
                along_x, along_y = 0, 0

            label_offset = 0.4 * (rad / abs(rad)) if rad != 0 else 0
            label_x = mid_x - label_offset * perp_x
            label_y = mid_y - label_offset * perp_y

            # Spread labels along the edge for each successive label
            if occurrence > 1:
                spread = 0.25 * (occurrence - 1)
                label_x += spread * along_x
                label_y += spread * along_y

            ax.annotate(f'g:{g1}\u2192{g2}\nt:{t1}\u2192{t2}', (label_x, label_y),
                        fontsize=7, ha='center', color=path_color,
                        fontweight='bold', backgroundcolor='white')

    # Highlight path nodes
    nx.draw_networkx_nodes(N, pos, nodelist=list(path_nodes), node_color='salmon',
                           node_size=600, ax=ax)

    return path_edges, path_nodes

# Detect charges: arcs where physical node stays same but battery changes
def find_charges(flow):
    charges = []
    for arc_key in flow.keys():
        src, dst = arc_key
        i, g1, t1, _ = tuple(src)
        j, g2, t2, _ = tuple(dst)
        if i == j and g1 != g2:
            charges.append({
                'node': i,
                'time': t1,
                'g_before': g1,
                'g_after': g2,
                't_after': t2
            })
    return charges

# Define colors
load_color = 'red'
ener_colors = ['blue', 'green', 'purple', 'orange', 'cyan']

# Decompose flows into paths
load_paths = flow_decomposition(x_load)
ener_paths = flow_decomposition(x_ener)

# Plot x_load paths
for idx, path in enumerate(load_paths):
    plot_path(path, load_color, rad=-0.2 - idx * 0.1)

# Plot x_ener paths
ener_rads = [0.2, 0.4, 0.6, 0.8, 1.0]
for idx, path in enumerate(ener_paths):
    color = ener_colors[idx % len(ener_colors)]
    rad = ener_rads[idx % len(ener_rads)]
    plot_path(path, color, rad)

    # Add truck icon at the start
    if path:
        start_node = tuple(path[0][0])[0]
        x, y = pos[start_node]
        ax.plot(x - 0.15, y + 0.12, marker='>', markersize=10, color=color,
                markeredgecolor='black', markeredgewidth=1, zorder=10)

# Annotate charges
charges = find_charges(x_load)
for charge in charges:
    node = charge['node']
    x, y = pos[node]
    star_x = x + 0.15
    star_y = y + 0.08
    ax.plot(star_x, star_y, marker='*', markersize=18, color='gold', markeredgecolor='black',
            markeredgewidth=2, zorder=10)
    ax.annotate(f"CHARGE @ t={charge['time']}: g {charge['g_before']}\u2192{charge['g_after']}",
                (star_x, star_y), xytext=(x + 0.5, y + 0.35),
                fontsize=9, ha='left', color='red', fontweight='bold',
                backgroundcolor='lightyellow',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Legend
legend_elements = [
    Line2D([0], [0], color=load_color, linewidth=3, label=f'x_load ({len(load_paths)} path(s))'),
]
for idx in range(len(ener_paths)):
    legend_elements.append(Line2D([0], [0], color=ener_colors[idx % len(ener_colors)],
                                  linewidth=3, label=f'x_ener Path {idx+1}'))
legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markeredgecolor='black', markersize=15, label='Charge location'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title(f'Heuristic-Tractor Solution (dee318c0)\n'
             f'x_load: {len(load_paths)} path(s), x_ener: {len(ener_paths)} path(s)\n'
             f'Objective: {data["objective"]:.4f}')
plt.tight_layout()
plt.savefig('flow_plot_dee318c0_heuristic_tractor.png', dpi=150)
plt.show()
