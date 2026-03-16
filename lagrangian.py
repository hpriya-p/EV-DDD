from Instance_v2 import Instance
from ChargeTimeNetwork_v2 import ChargeTimeNetwork
import gurobipy as gp
import pymetis
import networkx as nx




def partition_graph(N, k):
    """
    Compute a k-balanced spatial partition of N and a 2-D partition of Nt into
    subgraphs of Nt.

    Spatial step
    ------------
    N is partitioned into k balanced parts G_1, ..., G_k using METIS.

    Time-expanded step
    ------------------
    Nt has nodes (i, t) where i is a node of N and t is a discrete time index.
    T is inferred as max(t) + 1 over all nodes of Nt.  The time axis is divided
    into ceil(T / T_delta) contiguous blocks of width T_delta.  For every pair
    (l, p) with l in 0..k-1 and p in 0..n_blocks-1, the corresponding block of Nt
    is the node set

        { (i, t) : i in G_l  and  T_delta * p <= t < T_delta * (p + 1) }

    A node-induced subgraph of Nt is built for each block, augmented with fake
    boundary nodes for edges that cross into or out of the block.

    Parameters
    ----------
    N       : networkx.Graph or networkx.DiGraph
    Nt      : networkx.DiGraph
        Time-expanded network.  Nodes must be 2-tuples (i, t).
    k       : int
        Number of spatial partitions.
    T_delta : int
        Width of each time block.

    Returns
    -------
    subgraphs    : list of lists of networkx.DiGraph  (shape k x n_blocks)
        subgraphs[l][p] is the node-induced subgraph of Nt for block (l, p),
        augmented with fake s/t boundary nodes for cross-block edges.
    part_vec     : list of int (length len(N))
        part_vec[idx] is the spatial partition index for the idx-th node of N.
    cut_count    : int
        Number of spatial edges cut by the partition.
    """
    import math

    nodes = list(N.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # ── Spatial partition via METIS ──────────────────────────────────────────
    adjacency = [[] for _ in nodes]
    for u, v in N.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adjacency[i].append(j)

    cut_count, part_vec = pymetis.part_graph(k, adjacency=adjacency)
    node_to_part = {n: part_vec[i] for i, n in enumerate(nodes)}

    return cut_count, node_to_part, [e for e in N.edges if node_to_part[e[0]] != node_to_part[e[1]]]



    # ── Assign each Nt node to a block (l, p) ───────────────────────────────
    T = max(t for (_, t) in Nt.nodes) + 1
    n_blocks = math.ceil(T / T_delta)

    # node_block[(i, t)] = (l, p)
    node_block = {}
    for (i, t) in Nt.nodes:
        if i not in node_to_part:
            continue
        l = node_to_part[i]
        p = t // T_delta
        node_block[(i, t)] = (l, p)

    # ── Build one subgraph of Nt per block, with fake boundary nodes ─────────
    subgraphs = [[nx.DiGraph() for _ in range(n_blocks)] for _ in range(k)]

    # Add all internal nodes first
    for (i, t), (l, p) in node_block.items():
        subgraphs[l][p].add_node((i, t))

    # Add edges: internal edges are added as-is; cross-block edges get a fake
    # boundary node so that each subgraph remains self-contained.
    for u, v, data in Nt.edges(data=True):
        block_u = node_block.get(u)
        block_v = node_block.get(v)
        if block_u is None or block_v is None:
            continue
        if block_u == block_v:
            l, p = block_u
            subgraphs[l][p].add_edge(u, v, **data)
        else:
            # Outgoing boundary: u's subgraph gets a fake sink node
            lu, pu = block_u
            subgraphs[lu][pu].add_edge(u, f"t_{lu}_{pu}_{v}", **data)
            # Incoming boundary: v's subgraph gets a fake source node
            lv, pv = block_v
            subgraphs[lv][pv].add_edge(f"s_{lv}_{pv}_{u}", v, **data)

    return subgraphs, part_vec, cut_count




def lagrangian_decomposition(N, K, T_delta, parameters, config=['heuristic']):
    """
    Create subproblem instances for Lagrangian decomposition.

    For each subgraph in L, constructs an Instance and augments its objective
    with Lagrangian penalty terms for boundary edges incident to special nodes S:

        obj += - sum_{e=(u,v): u in N.nodes, v in S}  (lambda1[e]*x_load[e] + lambda2[e]*x_ener[e])
               + sum_{e=(u,v): u in S,       v in N.nodes} (lambda1[e]*x_load[e] + lambda2[e]*x_ener[e])

    Parameters
    ----------
    L        : list of subgraphs of N
                   Subgraphs of the full network N; one Instance is built per entry.
    S        : collection
                   Special (boundary) original-node IDs shared across subgraphs.
    lambda1  : dict  {augmented_edge -> float}
                   Multipliers for x_load on boundary edges.
    lambda2  : dict  {augmented_edge -> float}
                   Multipliers for x_ener on boundary edges.
    parameters : dict
                   Problem parameters forwarded to each Instance constructor.
    config   : list, optional
                   Configuration flags forwarded to each Instance constructor.

    Returns
    -------
    instances : list of Instance
        One Instance per subgraph in L with modified Lagrangian objectives.
    """
    instances = dict()
    cut_count, partition, boundary_edges = partition_graph(N, K)
    times = list(range(0, parameters['T'], T_delta))

    # construct relevant instances 
    for k in range(K):
        for min_t in times:
            params = {(key, val) for key, val in parameters.items()}
            params['min_time'] = min_t
            params['T'] = min(min_t + T_delta, parameters['T'])
            Vk = [i for i in N.nodes if partition[i] == k]
            Nk = N.subgraph(Vk).copy()

            # handle boundary edges due to spatial location
            in_edges = [e for e in boundary_edges if e[0] not in Vk and e[1] in Vk]
            out_edges = [e for e in boundary_edges if e[0] in Vk and e[1] not in Vk ]
            Nk.add_edges_from(in_edges + out_edges)
            srcs = [e[0] for e in in_edges]
            sinks = [e[1] for e in out_edges]
            params['sources'] += srcs
            params['sinks'] += sinks

    
            # Upshot of heuristic simplification: time-based boundary edges are waiting edges of the form (i, min_time - 1),(i, min_time) 

            params['bndry_wait_edges'] = Vk

            instances[(k, min_t)] = Instance(Nk, params, config)
         
    def get_boundary_edges(Ntl, partition_id, min_time):
        in_edges += [e for e in Ntl.edges if partition.get(e[0][0], -1) != partition_id]
        out_edges += [e for e in Ntl.edges if partition.get(e[1][0], -1) != partition_id]
        return in_edges, out_edges
  

    solutions = dict()
    properties = dict()
    values = dict()
    n_iter = 0
    lagrangified_edges = dict()
    for part, inst in instances.items():
        k, min_t = part
        soln, val, prop = inst.run_DDD()
        if 'x_load' not in solutions.keys():
            solutions['x_load'] = soln['x_load']
        else:
            solutions['x_load'].update(soln['x_load'])

        if 'x_ener' not in solutions.keys():
            solutions['x_ener'] = soln['x_ener']
        else:
            solutions['x_ener'].update(soln['x_ener'])

        if 'a' not in solutions.keys():
            solutions['a'] = soln['a']
        else:
            solutions['a'].update(soln['a'])

        if 'n' not in solutions.keys():
            solutions['n'] = soln['n']
        else:
            solutions['n'].update(soln['n'])

        properties[(n_iter, k, min_t)] = prop

        bndry = get_boundary_edges(inst.Ntl.Ntl, k, min_t)
        for e in bndry:
            if e[0][0] == f"s_{e[1][0]}":
                i, g, t, q = e[1]
                v1 = (i, g, t-1, q)
                v2 = e[1]
                lagrangified_edges[e] = (v1, v2)
            elif e[1][0] == f"t_{e[0][0]}":
                i, g, t, q = e[0]
                v2 = (i, g, t+1, q)
                v1 = e[0]
                lagrangified_edges[e] = (v1, v2)
            else:
                u, v = e
                lagrangified_edges[e] = e


    for e, matching_e in lagrangified_edges.items():
        if len(in_out['in']) == 0 and len(in_out['out']) == 0:
            continue 
        assert len(in_out['in']) == 2
        assert len(in_out['out']) == 2
        lambda1[e] = lambda1.get(e, 0) + lambda1()
    n_iter += 1




    return instances


def construct_time_expanded_network(N, T):
    nodes = [(i, t) for i in N.nodes for t in range(T)]
    Nt = nx.DiGraph()

    for i, t in nodes:
        Nt.add_edges_from(((i, t), (j, int(t + N[i][j]['time']))) for j in N.neighbors(i) if t + N[i][j]['time'] < T)

    return Nt