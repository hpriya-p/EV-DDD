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
    def __init__(self, N, parameters, init_L, init_T):
        self.N = N
        self.Ntl = nx.MultiDiGraph()
        self.param = parameters
        self.charges = init_L
        self.times = init_T
        self.edge_types = dict() #keys are edges ((i, j), \delta_g)
        self.edge_times = dict()
        self.edge_dists = dict()
        self.edge_charges = dict()
       
        self.construct()
        
    
    def construct(self):
        for i in tqdm(self.N.nodes):
            for t in self.times[i]:
                if t == self.param['T'] - 1:
                    continue
                if i in self.param['battery_nodes'] or i in self.param['tractor_nodes']:
                    self.__add_swap_edges(i, t)
                if i in self.param['charge_nodes']:
                    self.__add_charge_edges(i, t)
                self.__add_transit_edges(i, t)
            for g in self.charges[i]:
                self.__add_wait_edges_g(i,g)

    
    def manual_check_swap(self, v1, v2):
        """
        e: edge 
        """
        i, g1, t1 = v1
        j, g2, t2 = v2
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
        result = max_val # self.param['L']
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
        v1: (i, g1, t1)
        v2: (j, g2, t2)

        Corrects all edges between v1 and v2 in Ntl
        """

        i, g1, t1 = v1
        j, g2, t2 = v2
        types = [v for k, v in self.edge_types.items() if k[0] == v1 and k[1] == v2]
        for e_type in types:
            self.add_single_edge(v1, v2, e_type)
        
        # delete all copies of incorrect edges
        while (v1, v2, 0) in self.Ntl.edges:
            self.Ntl.remove_edge(v1, v2)  # remove incorrect edge

    def add_single_edge(self, v1, v2, e_type):
        """
        v1: (i, g1, t1)
        v2: (j, g2, t2)
        e_type: one of 'transit', 'charge', 'wait', 'charge', 'recourse'

        """
        i, g1, t1 = v1
        j, g2, t2 = v2
        assert g1 <= self.param['L'] and t1 < self.param['T']
        assert g2 <= self.param['L'] and t2 < self.param['T']
        g = self.__overestimate_val(self.charges[j], g2, self.param['L'])

        if t2 >= self.param['T']:
            t = self.param['T'] - 1
        else:
            t = self.__underestimate_val(self.times[j], t2)

        if ((i, g1, t1), (j, g, t), 0) in self.Ntl.edges and e_type in [v for k, v in self.edge_types.items() if k[0] == v1 and k[1] == (j, g, t)]:
            return  [k for k, v in self.edge_types.items() if k[0] == v1 and k[1] == (j, g, t) and v == e_type][0]

        # # check if the new edge is a true edge that is already present in the network:
        # if g2 == g and t2 == t:
        #     matching_edges = [e for e in self.Ntl.edges if e[0] == v1 and e[1] == v2 and self.edge_charges[e] == max(0, g2 - g1) and self.edge_dists[e] == max(g1 - g2, 0) and self.edge_times[e] == t2 - t1]
        #     if len(matching_edges) > 0:
        #         return matching_edges[0]

        
        key = self.Ntl.add_edge(v1, (j, g, t))
        new_e = (v1, (j, g, t), key)


        if e_type == 'transit_L' or e_type == 'transit_H':
            self.edge_charges[new_e] = 0
            self.edge_times[new_e] = t2 - t1
            self.edge_dists[new_e] = g1 - g2
            self.edge_types[new_e] = e_type

        elif e_type == 'wait':
            self.edge_charges[new_e] = 0
            if t2 != self.param['T']-1:
                self.edge_times[new_e] = t2 - t1
            else:
                self.edge_times[new_e] = 0
            self.edge_dists[new_e] = g1 - g2
            self.edge_types[new_e] = e_type
            

        elif e_type == 'charge' or e_type == 'swap':
            self.edge_charges[new_e] = g2 - g1
            self.edge_times[new_e] = t2 - t1
            self.edge_dists[new_e] = 0
            self.edge_types[new_e] = e_type

        
        
        return new_e

        
    # Helper functions for network construction 
    
    def add_out_edges(self, i, g, t):
        edges = []
        for j in self.N.neighbors(i):
            if self.N[i][j]['dH'] <= g and t + self.N[i][j]['time'] < self.param['T']:
                e = self.add_single_edge((i, g, t), (j, g - self.N[i][j]['dH'], t + self.N[i][j]['time']), 'transit_H')
                edges.append(e)      
            if self.N[i][j]['dL'] <= g and t + self.N[i][j]['time'] < self.param['T']:
                e = self.add_single_edge((i, g, t), (j, g - self.N[i][j]['dL'], t + self.N[i][j]['time']), 'transit_L')
                edges.append(e)   
        
        t_index = self.times[i].index(t)
        if t_index < len(self.times[i]) - 1:
            t_next = self.times[i][t_index + 1]
            e = self.add_single_edge((i, g, t), (i, g, t_next), 'wait')
            edges.append(e)

        for g2 in range(g + 1, g + self.param['charge_rate'][i] + 1):
            if g2 <= self.param['L'] and t + 1 < self.param['T']:
                e = self.add_single_edge((i, g, t), (i, g2, t + 1), 'charge')
                edges.append(e)
        
        return edges

    def __add_transit_edges(self, i, g_or_t, time=True):
        """
        Adds transit edges at station i and time g_or_t (if time=True); otherwise adds transit edges at station i and charge g (if time=False)
        
        Note: if the high-weight transit edge and low-weight transit edge both map to the same edge in the charge augmented network, the edge should be mapped as a high weight edge
        """
        edges = []
        if time:
            for g in self.charges[i]:
                for j in self.N.neighbors(i):
                    if self.N[i][j]['dH'] <= g and g_or_t + self.N[i][j]['time'] < self.param['T']:
                        e = self.add_single_edge((i, g, g_or_t), (j, g - self.N[i][j]['dH'], g_or_t + self.N[i][j]['time']), 'transit_H')
                        edges.append(e) 

                    if self.N[i][j]['dL'] <= g and g_or_t + self.N[i][j]['time'] < self.param['T']:
                        e = self.add_single_edge((i, g, g_or_t), (j, g - self.N[i][j]['dL'], g_or_t + self.N[i][j]['time']), 'transit_L')
                        edges.append(e)
        else:
            for t in self.times[i]:
                for j in self.N.neighbors(i):
                    if self.N[i][j]['dH'] <= g_or_t and t + self.N[i][j]['time'] < self.param['T']:
                        e = self.add_single_edge((i, g_or_t, t), (j, g_or_t - self.N[i][j]['dH'], t + self.N[i][j]['time']), 'transit_H')
                        edges.append(e)
                        
                    if self.N[i][j]['dL'] <= g_or_t and t + self.N[i][j]['time'] < self.param['T']:
                        e = self.add_single_edge((i, g_or_t, t), (j, g_or_t - self.N[i][j]['dL'], t + self.N[i][j]['time']), 'transit_L')
                        edges.append(e)

                
                        
        return edges

    def __add_wait_edges_g(self, i, g):
        """
        Adds waiting edges at node i and charge g
        """ 
        edges = []
        for l in range(len(self.times[i]) - 1):
            t = self.times[i][l]
            t_next = self.times[i][l + 1]
            e = self.add_single_edge((i, g, t), (i, g, t_next), 'wait')

            edges.append(e)
        
            if t_next < self.param['T'] - 1:
                e = self.add_single_edge((i, g, t), (i, g, self.param['T'] - 1), 'wait')
            edges.append(e)

            
        return edges
    
    def __add_charge_edges(self, i, g_or_t, time=True):
        """
        Adds charge edges at station i and time g_or_t (if time=True); otherwise adds charge edges at station i and charge g (if time=False)
        """
        edges = []
        if time:
            for g in self.charges[i]:
                for g2 in range(g + 1, g + self.param['charge_rate'][i] + 1):
                    if g2 <= self.param['L'] and g_or_t + 1 < self.param['T']:
                        e = self.add_single_edge((i, g, g_or_t), (i, g2, g_or_t + 1), 'charge')
                        edges.append(e)
                    
        else:
            for t in self.times[i]:
                for g2 in range(g_or_t + 1, g_or_t + self.param['charge_rate'][i] + 1):
                    if g2 <= self.param['L'] and  t + 1 < self.param['T']:
                        e = self.add_single_edge((i, g_or_t, t), (i, g2, t + 1), 'charge')
                        edges.append(e) 
        return edges
    
    
    def __add_swap_edges(self, i, g_or_t, time=True):
        """
        Adds swapping edges at battery/tractor swap station i and time g_or_t
        """
        edges = []

        if time:
            if g_or_t + self.param['bat_swap_time'] >= self.param['T']:
                return edges
            for g in self.charges[i]:
                for g2 in self.charges[i]:
                    if g2 <= g:
                        continue
                    if i in self.param['battery_nodes'] and g_or_t + self.param['bat_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g, g_or_t), (i, g2, g_or_t + self.param['bat_swap_time']), 'swap')
                    elif i in self.param['tractor_nodes'] and g_or_t + self.param['tr_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g, g_or_t), (i, g2, g_or_t + self.param['tr_swap_time']), 'swap')
                    else:
                        return edges
                    edges.append(e)
                    
        else:
            for t in self.times[i]:
               for g2 in self.charges[i]:
                    if g2 <= g_or_t:
                        continue
                    if i in self.param['battery_nodes'] and t + self.param['bat_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g_or_t, t), (i, g2, t + self.param['bat_swap_time']), 'swap')
                    elif i in self.param['tractor_nodes'] and t + self.param['tr_swap_time'] < self.param['T']:
                        e = self.add_single_edge((i, g_or_t, t), (i, g2, t + self.param['tr_swap_time']), 'swap')
                    else:
                        return edges
    
                    edges.append(e)
                    
        return edges


    """
     Flow decomposition and others
    
    """


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
                    adj.setdefault(e[0], []).append((e[1],e[2]))
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
            i, g, t = curr
            if t >= self.param['T'] - 1:
                break
            j = v2[0]
            g2 = g - self.edge_dists[e] + self.edge_charges[e]
            t2 = min(t + self.edge_times[e], self.param['T'] - 1)
            if t2 > self.param['T'] - 1 or g2 < 0:
                e2 = (i, g, t), (i, g, self.param['T'] - 1)
                new_P.append(e2)
                e_types[e2] = 'wait'
                edge_map[e2] = e
                return new_P, e_types, edge_map, False

                # charging_time = int(np.ceil(abs(g2)/self.param['charge_rate'][i]))
                # v_charged = (i, g + abs(g2), t + charging_time)
                # new_P.append((curr, v_charged))
                # e_types[(curr, v_charged)] = 'charge'
                # g2 = 0
                # t2 = min(t + charging_time + self.edge_times[e], self.param['T'] - 1)
                # new_P.append((v_charged,(j, g2, t2)))
                # e_types[(v_charged, (j, g2, t2))] = self.edge_types[e]
                # edge_map[(v_charged, (j, g2, t2))] = e
            else:
                new_P.append((curr, (j, g2, t2)))
                e_types[(curr, (j, g2, t2))] = self.edge_types[e] 
                edge_map[(curr,(j, g2, t2))] = e
            curr = (j, g2, t2)

        if len(P) == 1:
            new_P.append((curr, (curr[0], curr[1], self.param['T'] - 1)))
            e_types[(curr, (curr[0], curr[1], self.param['T'] - 1))] = 'wait'
        elif t < self.param['T'] - 1:
            new_P.append((curr, (j, g2, self.param['T'] - 1)))
            e_types[(curr, (j, g2, self.param['T'] - 1))] = 'wait'
        return new_P, e_types, edge_map, True 
    
    def convert_DDD_path(self, P, edge_types):
        """
        P: sequence of NODES in Ntl 

        Returns correct path (sequence of EDGES)
        """
        new_P = []
        curr = P[0]
        for l in range(len(P) - 1):
            e = (P[l], P[l+1]) 
            v1, v2 = e
            i, g1, t1 = v1
            j, g2, t2 = v2
            __, g_curr, t_curr = curr
            
            if g2 != self.param['L']:
                g_next = min(g_curr + (g2 - g1), self.param['L'])
            else:
                g_next = self.param['L']

            if t2 != self.param['T'] - 1:
                t_next = min(t_curr + (t2 - t1), self.param['T'] - 1)
            else:
                t_next = self.param['T'] - 1
            new_e = self.add_single_edge(curr, (j, g_next, t_next), edge_types[e])
            new_P.append(new_e)
            curr = new_e[1]
        return new_P

    # Update graph


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

                        return new_edge, del_e
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
                i, g1, t1 = v1
                j, g2, t2 = v2
                if g1 not in self.charges[i]:
                    self.charges[i].append(g1)
                    self.charges[i].sort()  
                if g2 not in self.charges[j]:  
                    self.charges[j].append(g2)
                    self.charges[j].sort()
                if t1 not in self.times[i]:
                    self.times[i].append(t1)
                    self.times[i].sort()
                if t2 not in self.times[j]:
                    self.times[j].append(t2)
                    self.times[j].sort()


                new_e = self.add_single_edge(v1, v2, edge_types[e])
                # there may be multiple edges: correct all of them
                self.correct_edge(edge_map[e][0], edge_map[e][1])
                return new_e, edge_map[e] 

        return None, None
        

    