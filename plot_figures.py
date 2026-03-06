import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

# Your flow data
x_load = {
    ((1, 5, 0), (3, 3, 4)): 10.0,
    ((3, 3, 4), (3, 5, 6)): 10.0,      # Node 3 -> 3 (charging/waiting)
    ((3, 5, 6), (12, 3, 10)): 10.0,
    ((12, 3, 10), (13, 2, 13)): 10.0,
    ((13, 2, 13), (24, 0, 17)): 10.0,
    ((24, 0, 17), (24, 0, 99)): 10.0   # Node 24 -> 24 (sink)
} 
x_ener =  {((1, 5, 0), (3, 3, 4)): 10.0, ((3, 3, 4), (3, 3, 99)): 10.0, ((3, 5, 0), (3, 5, 1)): 10.0, ((3, 5, 1), (3, 5, 4)): 10.0, ((3, 5, 4), (3, 5, 6)): 10.0, ((3, 5, 6), (12, 3, 10)): 10.0, ((12, 3, 10), (13, 2, 13)): 10.0, ((13, 2, 13), (24, 0, 17)): 10.0, ((24, 0, 17), (24, 0, 99)): 10.0}
a = {1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 4: 0.0, 12: 0.0, 5: 0.0, 11: 0.0, 9: 0.0, 8: 0.0, 7: 0.0, 18: 0.0, 16: 0.0, 10: 0.0, 15: 0.0, 17: 0.0, 14: 0.0, 13: 0.0, 24: 0.0, 23: 0.0, 19: 0.0, 22: 0.0, 20: 0.0, 21: 0.0}, 
n = {1: 0.0, 2: 0.0, 3: 10.0, 6: 0.0, 4: 0.0, 12: 0.0, 5: 0.0, 11: 0.0, 9: 0.0, 8: 0.0, 7: 0.0, 18: 0.0, 16: 0.0, 10: 0.0, 15: 0.0, 17: 0.0, 14: 0.0, 13: 0.0, 24: 0.0, 23: 0.0, 19: 0.0, 22: 0.0, 20: 0.0, 21: 0.0}



# Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Draw base network (light gray)
nx.draw_networkx_edges(N, pos, edge_color='lightgray', alpha=0.5, ax=ax)
nx.draw_networkx_nodes(N, pos, node_color='lightblue', node_size=500, ax=ax)
nx.draw_networkx_labels(N, pos, font_size=10, ax=ax)

# Decompose flow into paths
def flow_decomposition(flow):
    """Decompose a flow into distinct paths from sources to sinks."""
    # Build adjacency from flow arcs
    adj = {}  # node_state -> list of (next_node_state, arc_key)
    in_degree = {}
    out_degree = {}

    for arc_key in flow.keys():
        (i, g1, t1), (j, g2, t2) = arc_key
        src = (i, g1, t1)
        dst = (j, g2, t2)

        if src not in adj:
            adj[src] = []
        adj[src].append((dst, arc_key))

        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
        if src not in in_degree:
            in_degree[src] = 0
        if dst not in out_degree:
            out_degree[dst] = 0

    # Find source nodes (in_degree == 0)
    sources = [node for node in in_degree if in_degree[node] == 0]

    # Trace paths from each source
    paths = []
    for source in sources:
        path = []
        current = source
        while current in adj and adj[current]:
            next_node, arc_key = adj[current].pop(0)
            path.append(arc_key)
            current = next_node
        if path:
            paths.append(path)

    return paths

# Plot a single path
def plot_path(path, path_color, rad):
    """
    Plot a single path with specified color and curvature.
    """
    path_edges = [(i, j) for (i, g1, t1), (j, g2, t2) in path if i != j]
    path_nodes = set()
    for (i, g1, t1), (j, g2, t2) in path:
        path_nodes.add(i)
        path_nodes.add(j)

    # Draw edges
    nx.draw_networkx_edges(N, pos, edgelist=path_edges, edge_color=path_color,
                           width=3, arrows=True, arrowsize=20, ax=ax,
                           connectionstyle=f"arc3,rad={rad}")

    # Add annotations for battery levels and time at the middle of curved arcs
    for (i, g1, t1), (j, g2, t2) in path:
        if i != j:
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            # Calculate midpoint of the curved arc
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Calculate perpendicular offset for the curve
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                # Perpendicular unit vector (rotated 90 degrees)
                perp_x = -dy / length
                perp_y = dx / length
                # For arc3, the curve offset at midpoint
                curve_offset = rad * length * 0.25
                arc_mid_x = mid_x + perp_x * curve_offset
                arc_mid_y = mid_y + perp_y * curve_offset
            else:
                arc_mid_x, arc_mid_y = mid_x, mid_y

            # Position label along the arc direction based on rad sign
            label_offset = 0.4 * (rad / abs(rad)) if rad != 0 else 0
            label_x = mid_x - label_offset * perp_x
            label_y = mid_y - label_offset * perp_y
            ax.annotate(f'g:{g1}→{g2}\nt:{t1}→{t2}', (label_x, label_y),
                        fontsize=8, ha='center', color=path_color,
                        fontweight='bold', backgroundcolor='white')

    # Highlight path nodes
    nx.draw_networkx_nodes(N, pos, nodelist=list(path_nodes), node_color='salmon',
                           node_size=600, ax=ax)

    return path_edges, path_nodes
            
# Detect swaps: arcs in x_load where i == j and battery level changes (swap occurs)
def find_swaps(x_load):
    """Find swap events: same node, battery level changes."""
    swaps = []
    for (i, g1, t1), (j, g2, t2) in x_load.keys():
        if i == j and g1 != g2:  # Same node, battery changed = swap
            swaps.append({
                'node': i,
                'time': t1,
                'g_before': g1,
                'g_after': g2,
                't_after': t2
            })
    return swaps

# Define colors for each path
load_color = 'red'               # Red for x_load
ener_colors = ['blue', 'green']  # Blue and green for x_ener paths

# Decompose flows into paths
load_paths = flow_decomposition(x_load)
ener_paths = flow_decomposition(x_ener)

# Plot x_load path(s)
for idx, path in enumerate(load_paths):
    plot_path(path, load_color, rad=-0.2)

# Plot each x_ener path separately with different colors and curvatures
ener_rads = [0.2, 0.4]  # Different curvatures for each path
for idx, path in enumerate(ener_paths):
    color = ener_colors[idx % len(ener_colors)]
    rad = ener_rads[idx % len(ener_rads)]
    plot_path(path, color, rad)

    # Add truck icon at the start of each x_ener path
    if path:
        (start_node, g1, t1), _ = path[0]
        x, y = pos[start_node]
        # Use a truck-like marker (triangle)
        ax.plot(x - 0.15, y + 0.12, marker='>', markersize=10, color=color,
                markeredgecolor='black', markeredgewidth=1, zorder=10)

swaps = find_swaps(x_load)

# Annotate swap locations
for swap in swaps:
    node = swap['node']
    x, y = pos[node]
    # Add a star marker right beside the node
    star_x = x + 0.15
    star_y = y + 0.08
    ax.plot(star_x, star_y, marker='*', markersize=18, color='gold', markeredgecolor='black',
            markeredgewidth=2, zorder=10)
    # Add swap annotation closer to the star
    ax.annotate(f"SWAP @ t={swap['time']}: g {swap['g_before']}→{swap['g_after']}",
                (star_x, star_y), xytext=(x + 0.5, y + 0.35),
                fontsize=9, ha='left', color='red', fontweight='bold',
                backgroundcolor='lightyellow',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Add legend for flows
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

legend_elements = [
    Line2D([0], [0], color=load_color, linewidth=3, label=f'x_load (1 path)'),
]
for idx in range(len(ener_paths)):
    legend_elements.append(Line2D([0], [0], color=ener_colors[idx % len(ener_colors)],
                                  linewidth=3, label=f'x_ener Path {idx+1}'))
legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markeredgecolor='black', markersize=15, label='Swap location'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title(f'Flow Comparison: x_load (1 path) vs x_ener ({len(ener_paths)} paths)\n(labels show g: battery level, t: time)')
plt.tight_layout()
plt.savefig('flow_plot.png', dpi=150)
plt.show()
