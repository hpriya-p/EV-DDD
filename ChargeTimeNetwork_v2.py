import networkx as nx
import numpy as np
from collections import deque
from copy import deepcopy

from tqdm import tqdm


class RangeConstrViolation(Exception):
        """A custom exception for specific error scenarios."""
        def __init__(self, nodes, del_edges):
            self.affected_nodes = nodes
            self.del_edges = del_edges


class ChargeTimeNetwork:
    def __init__(self, N, parameters, config):
        self.N = N
        self.Ntl = nx.MultiDiGraph()
        self.param = parameters
        self.config = config
        self.edge_types = dict()  # keys are edges ((i, j), \delta_g)
        self.edge_times = dict()
        self.edge_dists = dict()
        self.edge_charges = dict()
        self.updated_since_refresh = set()

        if self.config == 'default':
            self.charges = {i: [0] + list(range(1, self.param['L'], self.param['step_size'])) + [self.param['L']] for i in self.N.nodes}
            self.charges = {i: list(np.unique(self.charges[i])) for i in self.charges.keys()}
            self.times = {i: [0] + list(range(1, self.param['T'] - 1, self.param['step_size'])) + [self.param['T'] - 1] for i in self.N.nodes}
            self.times = {i: list(np.unique(self.times[i])) for i in self.times.keys()}
        elif self.config == 'heuristic':
            thresholds = [self.param['speed_curve'][q]['maxbat'] for q in sorted(self.param['speed_curve'].keys())]
            self.charges = {i : sorted([0] + [L - self.N[i][j]['dH'] for L in thresholds for j in self.N.neighbors(i) if self.N[i][j]['dH'] < L] + thresholds) for i in self.N.nodes}
            self.charges = {i: list(np.unique(self.charges[i])) for i in self.charges.keys()}
            self.times = {i: [0] + list(range(1, self.param['T'] - 1, self.param['step_size'])) + [self.param['T'] - 1] for i in self.N.nodes}
            self.times = {i: list(np.unique(self.times[i])) for i in self.times.keys()}
            print(self.charges)
        else:
            raise ValueError("Invalid config")
        
        self.construct()


    def diff(self, other):
        
        equal = True

        if self.charges != other.charges:
            diff = {i for i in set(self.charges) | set(other.charges) if self.charges.get(i) != other.charges.get(i)}
            l = min(len(diff), 10)
            print(f"charges differ at nodes: {list(diff)[:l]}")
            equal = False

        if self.times != other.times:
            diff = {i for i in set(self.times) | set(other.times) if self.times.get(i) != other.times.get(i)}
            l = min(len(diff), 10)
            print(f"times differ at nodes: {list(diff)[:l]}")
            equal = False

        self_nodes = set(self.Ntl.nodes())
        other_nodes = set(other.Ntl.nodes())
        if self_nodes != other_nodes:
            l = min(len(self_nodes - other_nodes), 10) 
            print(f"nodes only in self:  {list(self_nodes - other_nodes)[:l]}")
            l = min(len(other_nodes - self_nodes), 10)
            print(f"nodes only in other: {list(other_nodes - self_nodes)[:l]}")
            equal = False

        self_edges = set(self.Ntl.edges())
        other_edges = set(other.Ntl.edges())
        if self_edges != other_edges:
            l = min(len(self_edges - other_edges), 10) 
            print(f"edges only in self:  {list(self_edges - other_edges)[:l]}")
            l = min(len(other_edges - self_edges), 10)
            print(f"edges only in other: {list(other_edges - self_edges)[:l]}")
            equal = False

        for attr in ('edge_types', 'edge_times', 'edge_dists', 'edge_charges'):
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr)
            if self_attr != other_attr:
                only_self = {k: self_attr[k] for k in self_attr if k not in other_attr or self_attr[k] != other_attr[k]}
                only_other = {k: other_attr[k] for k in other_attr if k not in self_attr or self_attr[k] != other_attr[k]}
                l = min(len(only_self), 10)
                l2 = min(len(only_other), 10)
                print(f"{attr} differs — self only/changed: {list(only_self.items())[:l]}, other only/changed: {list(only_other.items())[:l2]}")
                equal = False

        return equal

    def construct(self):
        for i in tqdm(self.N.nodes):
            for t in self.times[i]:
                if t == self.param['T'] - 1:
                    continue
                if i in self.param['battery_nodes'] or i in self.param['tractor_nodes']:
                    self.__add_swap_edges(i, t)
                if i in self.param['charge_nodes']:
                    self.__add_charge_edges(i, t)
                    self.__add_piecewise_edges(i, t)
                self.__add_transit_edges(i, t)
            for g in self.charges[i]:
                self.__add_wait_edges_g(i, g)

    def refresh(self):
        print("Recomputing edges for nodes: ", self.updated_since_refresh)

        # Step 1: Remove ALL edges incident to any network node (i, g, t, q)
        # where i is in updated_since_refresh (both outgoing and incoming).
        to_remove = [e for e in self.Ntl.edges
                     if e[0][0] in self.updated_since_refresh
                     or e[1][0] in self.updated_since_refresh]
        self.Ntl.remove_edges_from(to_remove)

        # Step 2: Re-add all outgoing edges from updated nodes.
        for i in tqdm(self.updated_since_refresh):
            for t in self.times[i]:
                if t == self.param['T'] - 1:
                    continue
                if i in self.param['battery_nodes'] or i in self.param['tractor_nodes']:
                    self.__add_swap_edges(i, t)
                if i in self.param['charge_nodes']:
                    self.__add_charge_edges(i, t)
                    self.__add_piecewise_edges(i, t)
                self.__add_transit_edges(i, t)
            for g in self.charges[i]:
                self.__add_wait_edges_g(i, g)

        # Step 3: Re-add incoming transit edges from non-updated nodes.
        # add_single_edge skips edges that already exist (non-updated -> non-updated),
        # and freshly adds edges to updated nodes (removed in step 1 above).
        print("Recomputing incoming transit edges from non-updated nodes")
        non_updated = [n for n in self.N.nodes if n not in self.updated_since_refresh]
        for i in tqdm(non_updated):
            for t in self.times[i]:
                if t == self.param['T'] - 1:
                    continue
                self.__add_transit_edges(i, t)

        self.updated_since_refresh = set()
        self.cleanup()
        

    
    




    def manual_check_swap(self, v1, v2):
        """
        Check if edge is swap edge or not 
        """
        i, g1, t1, q1 = v1
        j, g2, t2, q2 = v2
        if g1 >= g2:
            return False
        if i in self.param['battery_nodes'] and t2 - t1 == self.param['bat_swap_time']:
            return True
        elif i in self.param['tractor_nodes'] and t2 - t1 == self.param['tr_swap_time']:
            return True

        return False

    def __overestimate_val(self, lst, val, max_val):
        """
        Given sorted list lst, returns the smallest element >= val.
        Returns None if no such element exists.
        """
        if val in lst:
            return val
        left, right = 0, len(lst) - 1
        result = max_val
        while left <= right:
            mid = (left + right) // 2
            if lst[mid] == val:
                return lst[mid]
            if lst[mid] >= val:
                result = lst[mid]
                right = mid - 1
            else:
                left = mid + 1
        return result

    def __underestimate_val(self, lst, val):
        """
        Given sorted list lst, returns the largest element <= val.
        Returns None if no such element exists.
        """
        if val in lst:
            return val
        left, right = 0, len(lst) - 1
        result = None
        while left <= right:
            mid = (left + right) // 2
            if lst[mid] == val:
                return lst[mid]
            if lst[mid] <= val:
                result = lst[mid]
                left = mid + 1
            else:
                right = mid - 1
        return result

    def correct_edge(self, v1, v2):
        """
        v1: (i, g1, t1, q1)
        v2: (j, g2, t2, q2)

        Corrects all edges between v1 and v2 in Ntl
        """

        i, g1, t1, q1 = v1
        j, g2, t2, q2 = v2
        types = [v for k, v in self.edge_types.items() if k[0] == v1 and k[1] == v2]
        for e_type in types:
            self.add_single_edge(v1, v2, e_type)

        # delete all copies of incorrect edges
        while self.Ntl.has_edge(v1, v2):
            self.Ntl.remove_edge(v1, v2)


    def add_single_edge(self, v1, v2, e_type):
        """
        v1: (i, g1, t1, q1)
        v2: (j, g2, t2, q2)
        e_type: one of 'transit', 'charge', 'wait', 'charge', 'recourse'

        """
        i, g1, t1, q1 = v1
        j, g2, t2, q2 = v2
        assert g1 <= self.param['L'] and t1 < self.param['T']
        assert g2 <= self.param['L'] and t2 < self.param['T']
        g = self.__overestimate_val(self.charges[j], g2, self.param['L'])

        if t2 >= self.param['T']:
            t = self.param['T'] - 1
        else:
            t = self.__underestimate_val(self.times[j], t2)

        q = self.get_q(j,g)

        if ((i, g1, t1, q1), (j, g, t, q), 0) in self.Ntl.edges and e_type in [v for k, v in self.edge_types.items() if k[0] == v1 and k[1] == (j, g, t, q)]:
            return [k for k, v in self.edge_types.items() if k[0] == v1 and k[1] == (j, g, t, q) and v == e_type][0]

        key = self.Ntl.add_edge(v1, (j, g, t, q))
        new_e = (v1, (j, g, t, q), key)

        if e_type == 'transit_L' or e_type == 'transit_H':
            self.edge_charges[new_e] = 0
            self.edge_times[new_e] = t2 - t1
            self.edge_dists[new_e] = g1 - g2
            self.edge_types[new_e] = e_type

        elif e_type == 'wait':
            self.edge_charges[new_e] = 0
            if t2 != self.param['T'] - 1:
                self.edge_times[new_e] = t2 - t1
            else:
                self.edge_times[new_e] = 0
            self.edge_dists[new_e] = 0
            self.edge_types[new_e] = e_type

        elif e_type == 'charge' or e_type == 'swap' or e_type == 'piecewise':
            self.edge_charges[new_e] = g2 - g1
            self.edge_times[new_e] = t2 - t1
            self.edge_dists[new_e] = 0
            self.edge_types[new_e] = e_type

        return new_e


    # Helper functions for network construction


    def __add_transit_edges(self, i, g_or_t, time=True):
        """
        Adds transit edges at station i and time g_or_t (if time=True); otherwise adds transit edges at station i and charge g (if time=False)

        Note: if the high-weight transit edge and low-weight transit edge both map to the same edge in the charge augmented network, both multiedges are created
        """
        edges = []
        if time:
            for g in self.charges[i]:
                for j in self.N.neighbors(i):
                    if self.N[i][j]['dH'] <= g and g_or_t + self.N[i][j]['time'] < self.param['T']:
                        q1 = self.get_q(i, g)
                        q2 = self.get_q(j, g - self.N[i][j]['dH'])
                        e = self.add_single_edge((i, g, g_or_t, q1), (j, g - self.N[i][j]['dH'], g_or_t + self.N[i][j]['time'], q2), 'transit_H')
                        edges.append(e)

                    if self.N[i][j]['dL'] <= g and g_or_t + self.N[i][j]['time'] < self.param['T']:
                        q1 = self.get_q(i, g)
                        q2 = self.get_q(j, g - self.N[i][j]['dL'])
                        e = self.add_single_edge((i, g, g_or_t, q1), (j, g - self.N[i][j]['dL'], g_or_t + self.N[i][j]['time'], q2), 'transit_L')
                        edges.append(e)
        else:
            for t in self.times[i]:
                for j in self.N.neighbors(i):
                    if self.N[i][j]['dH'] <= g_or_t and t + self.N[i][j]['time'] < self.param['T']:
                        q1 = self.get_q(i, g_or_t)
                        q2 = self.get_q(j, g_or_t - self.N[i][j]['dH'])
                        e = self.add_single_edge((i, g_or_t, t, q1), (j, g_or_t - self.N[i][j]['dH'], t + self.N[i][j]['time'], q2), 'transit_H')
                        edges.append(e)

                    if self.N[i][j]['dL'] <= g_or_t and t + self.N[i][j]['time'] < self.param['T']:
                        q1 = self.get_q(i, g_or_t)
                        q2 = self.get_q(j, g_or_t - self.N[i][j]['dL'])
                        e = self.add_single_edge((i, g_or_t, t, q1), (j, g_or_t - self.N[i][j]['dL'], t + self.N[i][j]['time'], q2), 'transit_L')
                        edges.append(e)


        return edges

    def __add_wait_edges_g(self, i, g):
        """
        Adds waiting edges at node i and charge g
        """
        edges = []
        q = self.get_q(i, g)
        for l in range(len(self.times[i]) - 1):
            t = self.times[i][l]
            t_next = self.times[i][l + 1]
            e = self.add_single_edge((i, g, t, q), (i, g, t_next, q), 'wait')

            edges.append(e)

            if t_next < self.param['T'] - 1:
                e = self.add_single_edge((i, g, t, q), (i, g, self.param['T'] - 1, q), 'wait')
            edges.append(e)


        return edges

    def __add_charge_edges(self, i, t):
        """
        Adds charge edges at station i and time g_or_t (if time=True); otherwise adds charge edges at station i and charge g (if time=False)
        """
        edges = []
        for g in self.charges[i]:
            q1 = self.get_q(i, g)
            UB = self.param['speed_curve'][q1]['maxbat']
            if self.config != 'heuristic':
                for g2 in range(g + 1, int(min(UB + 1, g + self.param['charge_rate'][i] * self.param['speed_curve'][q1]['speed'] + 1))):
                    if g2 <= self.param['L'] and t + 1 < self.param['T']:
                        q2 = self.get_q(i, g2)
                        e = self.add_single_edge((i, g, t, q1), (i, g2, t + 1, q2), 'charge')
                        edges.append(e)
            else:
                # only create arcs corresponding to charging to full or charging just enough
                for g2 in self.charges[i] + [self.N[i][j]['dH'] for j in self.N.neighbors(i)]:
                    if g2 <= g or g2 > UB:
                        continue 
                    q2 = q1
                    t2 = np.ceil((g2 - g) / self.param['charge_rate'][i] * self.param['speed_curve'][q1]['speed'])
                    if t + t2 < self.param['T']:
                        e = self.add_single_edge((i, g, t, q1), (i, g2, t + int(t2), q2), 'charge')
                        edges.append(e)

        return edges


    def __add_swap_edges(self, i,t):
        """
        Adds swapping edges at battery/tractor swap station i and time g_or_t
        """
        edges = []

        if t + self.param['bat_swap_time'] >= self.param['T']:
            return edges
        for g in self.charges[i]:
            q1 = self.get_q(i, g)
            if self.config != 'heuristic':
                for g2 in self.charges[i]:
                    if g2 <= g:
                        continue
                    q2 = self.get_q(i, g2)
                    if i in self.param['battery_nodes'] and t + self.param['bat_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g, t, q1), (i, g2, t + self.param['bat_swap_time'], q2), 'swap')
                    elif i in self.param['tractor_nodes'] and t + self.param['tr_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g, t, q1), (i, g2, t + self.param['tr_swap_time'], q2), 'swap')
                    else:
                        return edges
                    edges.append(e)
            else:
                # only create arcs corresponding to swapping to one of the breakpoints of the speed curve for battery. For tractor, L - d_ij arcs are added
                thresholds = [self.param['speed_curve'][q]['maxbat'] for q in sorted(self.param['speed_curve'].keys())]
                thresholds = [g2 for g2 in thresholds if g2 > g]
                if i in self.param['battery_nodes']:
                    for g2 in thresholds:
                        q2 = self.get_q(i, g2)
                        if t + self.param['bat_swap_time'] < self.param['T']:
                            e = self.add_single_edge((i, g, t, q1), (i, g2, t + self.param['bat_swap_time'], q2), 'swap')
                            edges.append(e)
                else:
                    for g2 in self.charges[i]:
                        if g2 <= g:
                            continue
                        q2 = self.get_q(i, g2)
                        t2 = self.param['tr_swap_time']
                        if t + t2 < self.param['T']:
                            e = self.add_single_edge((i, g, t, q1), (i, g2, t + int(t2), q2), 'swap')
                            edges.append(e)

       
        return edges


    """
     Flow decomposition and others

    """
    def get_q(self, i, g):
        for q in sorted(self.param['speed_curve'].keys()):
            if g < self.param['speed_curve'][q]['maxbat']:
                return q
        return max(self.param['speed_curve'].keys())

    def __add_piecewise_edges(self, i, t):
        """
        Adds edges of form (i, g, t, q1), (i, g, t, q2)
        """
        thresholds = [self.param['speed_curve'][q]['maxbat'] for q in sorted(self.param['speed_curve'].keys())]
        for q in sorted(self.param['speed_curve'].keys()):
            g = self.param['speed_curve'][q]['maxbat']
            if g < self.param['L']:
                e = self.add_single_edge((i, g, t, q), (i, g, t, q + 1), 'piecewise')
        
    def flow_decomposition(self, orig_flow):
        """
        Path is a list of edges
        """
        flow = deepcopy(orig_flow)
        decomposition = dict()

        def get_path():
            E = [e for e, val in flow.items() if val > 0]
            VE = [e[0] for e in E] + [e[1] for e in E]
            src_nodes = [v for v in self.Ntl.nodes if v[2] == 0 and v in VE]
            targets = [v for v in self.Ntl.nodes if v[2] == self.param['T'] - 1 and v in VE]

            # Build adjacency list
            adj = dict()
            MULTIEDGE = False
            for e in E:
                if len(e) == 3:
                    adj.setdefault(e[0], []).append((e[1], e[2]))
                    MULTIEDGE = True
                else:
                    adj.setdefault(e[0], []).append(e[1])
            for s in src_nodes:
                for t in targets:
                    # BFS to find path from s to t
                    queue = deque()
                    queue.append((s, []))
                    visited = set()
                    visited.add(s)
                    while queue:
                        node, path = queue.popleft()
                        if node == t:
                            return path

                        if MULTIEDGE:
                            for neighbor_key in adj.get(node, []):
                                neighbor, key = neighbor_key
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, path + [(node, neighbor, key)]))
                        else:
                            for neighbor_key in adj.get(node, []):
                                neighbor = neighbor_key
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, path + [(node, neighbor)]))
            return None
        while True:
            P = get_path()
            if P is None:
                break
            min_cap = min(flow[e] for e in P)
            for e in P:
                flow[e] -= min_cap
            decomposition[tuple(P)] = min_cap

        return decomposition

    def convert_flow(self, lst_of_flows):
        corrected_flow = []
        overall_status = True
        for flow in lst_of_flows:
            mapped_flow = dict()
            for P, val in self.flow_decomposition(flow).items():
                actual_path, edge_types, e_maps, status = self.convert_actual_path(P)
                overall_status = overall_status and status
                for e in actual_path:
                    mapped_flow[e] = mapped_flow.get(e, 0) + val
            corrected_flow.append(mapped_flow)
        return corrected_flow, overall_status

    def convert_actual_path(self, P):
        """
        P: sequence of edges in Ntl

        Returns correct path, a sequence of edges
        """
        new_P = []
        e_types = dict()
        edge_map = dict()
        curr = P[0][0]

        for l in range(len(P) - 1):
            e = P[l]
            v1, v2, key = e
            i, g, t, q = curr
            if t >= self.param['T'] - 1:
                break
            j = v2[0]
            g2 = g - self.edge_dists[e] + self.edge_charges[e]
            t2 = min(t + self.edge_times[e], self.param['T'] - 1)
            q2 = self.get_q(j, g2)  
            if t2 > self.param['T'] - 1 or g2 < 0:
                e2 = (i, g, t, q), (i, g, self.param['T'] - 1, q)
                new_P.append(e2)
                e_types[e2] = 'wait'
                edge_map[e2] = e
                return new_P, e_types, edge_map, False

            else:
                new_P.append((curr, (j, g2, t2, q2)))
                e_types[(curr, (j, g2, t2, q2))] = self.edge_types[e]
                edge_map[(curr, (j, g2, t2, q2))] = e
            curr = (j, g2, t2, q2)

        if len(P) == 1:
            new_P.append((curr, (curr[0], curr[1], self.param['T'] - 1, curr[3])))
            e_types[(curr, (curr[0], curr[1], self.param['T'] - 1, curr[3]))] = 'wait'
        elif t < self.param['T'] - 1:
            new_P.append((curr, (j, g2, self.param['T'] - 1, q2)))
            e_types[(curr, (j, g2, self.param['T'] - 1, q2))] = 'wait'
        return new_P, e_types, edge_map, True
 
    def update(self, lst_of_flows):
        """
        Given flow vector, updates network. Returns new edges, new recourse edges, and deleted edges
        recomputes edges at affected nodes
        """
        for flow in lst_of_flows:
            for P, val in self.flow_decomposition(flow).items():
                if val > 0:
                    new_edge, del_e = self.__update_P(P)
                    if new_edge is not None:
                        # update edges incident to the affected nodes
                        i, j = new_edge[0][0], new_edge[1][0]
                        affected_edges = [e for e in self.Ntl.edges if e[0][0] == i or e[1][0] == i or e[0][0] == j or e[1][0] == j]
                        for e in affected_edges:
                            self.correct_edge(e[0], e[1])
                        self.cleanup()
                        return new_edge, del_e
        self.cleanup()
        return None, None

    def __update_P(self, P):
        """
        P: path that is list of edges

        Adds edges to Ntl in place, but does not delete them (returns the lst of edges to be deleted)
        """

        actual_path, edge_types, edge_map, status = self.convert_actual_path(P)

        for e in actual_path:
            if e not in self.Ntl.edges:
                v1, v2 = e
                i, g1, t1, q1 = v1
                j, g2, t2, q2 = v2
                if g1 not in self.charges[i]:
                    self.charges[i].append(g1)
                    self.charges[i].sort()
                    self.updated_since_refresh.add(i)
                if g2 not in self.charges[j]:
                    self.charges[j].append(g2)
                    self.charges[j].sort()
                    self.updated_since_refresh.add(j)
                if t1 not in self.times[i]:
                    self.times[i].append(t1)
                    self.times[i].sort()
                    self.updated_since_refresh.add(i)
                if t2 not in self.times[j]:
                    self.times[j].append(t2)
                    self.times[j].sort()
                    self.updated_since_refresh.add(j)

                new_e = self.add_single_edge(v1, v2, edge_types[e])
                return new_e, edge_map[e]

        return None, None

    def cleanup(self):
        # remove all keys that are not edges in Ntl from edge_types, edge_times, edge_dists, edge_charges
        edges = set(self.Ntl.edges)
        self.edge_types = {k: v for k, v in self.edge_types.items() if k in edges}
        self.edge_times = {k: v for k, v in self.edge_times.items() if k in edges}
        self.edge_dists = {k: v for k, v in self.edge_dists.items() if k in edges}
        self.edge_charges = {k: v for k, v in self.edge_charges.items() if k in edges} 