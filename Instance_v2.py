from ChargeTimeNetwork_v2 import *
from copy import deepcopy
import numpy as np
import gurobipy as gp
import time
import json
import signal
import datetime

PEN_CONST = 10**6


def _jsonable(obj):
    """Recursively make obj JSON-serializable, stringifying unsupported types and tuple keys."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (bool, type(None))):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


def read_ddd_log(filepath):
    """Load a log written by Instance.solve (see the 'ddd_log' parameter).

    Returns a list of records, one per line:
      {'event': 'start', ...}       run settings
      {'event': 'iteration', ...}   one per relaxation solved
      {'event': 'final', ...}       last record; holds the solution if DDD converged
    """
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


def discretization_from_log(filepath, iteration=None):
    """Time and charge discretization of one iteration of a ddd_log (default: the last).

    Returns {'iteration', 'times': {node: [t, ...]}, 'charges': {node: [g, ...]},
    'lower_bound', 'objective', 'start': the log's start record, 'path'}. The log is
    read line by line, so large logs are not loaded whole. A line cut off by a killed
    process is skipped.
    """
    start, chosen, final = None, None, None
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # the line being written when the process was killed
            if rec.get('event') == 'start' and start is None:
                start = rec
            elif rec.get('event') == 'iteration' and (iteration is None or rec['iteration'] == iteration):
                chosen = rec
            elif rec.get('event') == 'final':
                final = rec
    if chosen is None:
        raise ValueError(f"{filepath}: no iteration record" + (f" {iteration}" if iteration else ""))
    to_int = lambda d: {int(i): [int(v) for v in vals] for i, vals in d.items()}
    # Running time of the earlier run: up to its final record when it wrote one (stopped
    # runs do), which includes the iteration it was working on when it was stopped.
    elapsed = max((r.get('elapsed') or 0) for r in (chosen, final) if r is not None)
    return {'iteration': chosen['iteration'], 'times': to_int(chosen['times']),
            'charges': to_int(chosen['charges']), 'lower_bound': chosen.get('lower_bound'),
            'objective': chosen.get('objective'), 'elapsed': elapsed,
            'start': start or {}, 'path': filepath}


class Instance:
    def __init__(self, N, parameters, config=['heuristic'], network_only=False, seed=None):
        # parameters dict keys (includes all ChargeTimeNetwork keys, plus):
        #   L             (int)   – maximum battery level
        #   T             (int)   – time horizon; time steps run from min_time to T-1
        #   min_time      (int)   – earliest time step, default 0
        #   step_size     (int)   – discretisation stride for charge and time levels
        #   battery_nodes (list)  – nodes where battery swaps are available
        #   tractor_nodes (list)  – nodes where tractor swaps are available
        #   charge_nodes  (list)  – nodes where in-place charging is available
        #   bat_swap_time (int)   – time units consumed by a battery swap
        #   tr_swap_time  (int)   – time units consumed by a tractor swap
        #   charge_rate   (dict)  – {node: rate}; units of charge added per unit time at each charge node
        #   speed_curve   (dict)  – {q: {'speed': s, 'minbat': lo, 'maxbat': hi}};
        #                           piecewise-linear speed segments indexed by segment id q
        #   sources       (list)  – node IDs at which trucks originate (flow leaves at time min_time)
        #   sinks         (list)  – node IDs at which trucks terminate (flow arrives at time T-1)
        #   D             (dict)  – {(src, sink): val; number of trucks supplied by node and demanded at node. 
        #   N_tractors    (int)   – maximum number of tractors that can be placed at tractor nodes
        #   N_batteries   (int)   – maximum number of batteries that can be placed at battery nodes
        #   N_chargers    (int)   – maximum number of chargers that can be installed at charge nodes
        #   charge_cost   (dict)  – {(node, time): cost}; per-unit charge cost at node i at time t
        #   stat_cost     (dict)  – {node: cost}; cost per charger a_c[i] installed at node i
        #   site_install_cost (float) – (optional, default 0) cost of installing a swap site,
        #                           a_s[i] = 1, at a battery or tractor node i
        #   existing_charge_nodes (list) – (optional) existing charging stations: their current
        #                           capacity (charge_rate[i]) is available without installing,
        #                           i.e. charging <= Delta * charge_rate[i] * (a_c[i] + 1) there
        #   surplus_cost  (dict)  – {node: cost}; cost per surplus truck stationed at node
        #   MAX_ITER      (int)   – maximum number of column-generation / update iterations
        #   lagrange_multipliers (dict) – (optional) {'load': {edge: mu}, 'ener': {edge: mu}};
        #                                 Lagrange multipliers subtracted from the objective
        #   latest_origin_time (int) – (optional) latest time at which the load of an OD pair
        #                           may still be at its origin: for every commodity k with
        #                           origin o_k, load flow of k on any edge out of a node
        #                           (o_k, g, t) with t > latest_origin_time is zero (enforced
        #                           as upper bound 0 on those x_load variables). Unset: no limit
        #   initial_discretization (str or dict) – (optional) resume Bi-DDD from an earlier run:
        #                           the path of its ddd_log (its last iteration's times and
        #                           charges are added to the initial discretization), or a
        #                           dict {'times': {node: [...]}, 'charges': {node: [...]}}.
        #                           A log from a run with a different T, network type
        #                           (default/heuristic) or set of stations is refused
        #   fast_update   (dict)  – (optional) {'policy': 'simple_update', 'seconds': 18000}: use
        #                           this faster update policy until the solve's elapsed time
        #                           (counting earlier attempts of a resumed run) reaches
        #                           'seconds', then rebuild the network and model for the
        #                           current discretization and continue with the configured
        #                           policy. While the fast policy is in use the network need not
        #                           be a relaxation, so those iterations log their bound as
        #                           'relaxation_bound' and 'lower_bound' as None
        #   snapshot_on_interrupt (bool) – (optional, default False) on interrupt, write the
        #                           whole instance to instance_snapshot_<timestamp>.json in
        #                           the cwd (dump_to_json; hundreds of MB on large networks).
        #                           Resuming needs only the ddd_log
        #   ddd_log       (str)   – (optional) path of a JSON Lines file to which solve() writes
        #                           one record per DDD iteration and a final record with the
        #                           solution; read it back with read_ddd_log
        self.N = N
        self.param = parameters
        self.model = gp.Model() 
        self.VAR_ROOTS = ['x_load', 'x_ener']
        self.penalties = dict()
        self.config = config 
        if 'value_for_time' not in self.param.keys():
            self.param['value_for_time'] = 1

        if 'min_time' not in self.param.keys():
            self.param['min_time'] = 0
        # upate config
        if len(set(self.config) & set(['smart_update', 'simple_update', 'simple_w_incoming_update', 'local_rebuild_update'])) == 0:
            self.config = [x for x in self.config] + ['simple_w_incoming_update']

        # init charge-time-augmented network
        self.Ntl = ChargeTimeNetwork(N, self.param, config)
        # control printing via self.verbose
        self.verbose = self.param.get('verbose', True)
        self.resumed_from = None
        if self.param.get('initial_discretization') is not None:
            self.resumed_from = self._apply_initial_discretization(self.param['initial_discretization'])
        if self.verbose:
            print("Size of network: ", len(self.Ntl.Ntl.nodes), " nodes, ", len(self.Ntl.Ntl.edges), " edges")
        if seed is not None:
            # seed() was removed (it was faulty); warm-start with initial_discretization
            raise ValueError("Instance(seed=...) is no longer supported: pass the earlier run's "
                             "discretization as parameters['initial_discretization'] instead")
        if not network_only:
            self.construct_model()


        

    """
    Functions for model construction / update
    """
    
    def parse_var_name(self, var_name):
        # Assumes format: x_load[((i, g, t, q),(j, g', t', q'),(k)), l]
        inside = var_name[var_name.find('[')+1:var_name.find(']')]
        num_commas = inside.count(',')
        if num_commas == 0:
            i = int(inside.strip('()'))
            return i
        else:
            inside = inside.strip('[]')
            # Split commodity index l from the right: "((t1),(t2),(k)), l"
            inner, l_str = inside.rsplit(', ', 1)
            l = int(l_str)
            # Strip outer parens from the edge triple: "(t1),(t2),(k)"
            inner = inner.strip('()')
            tuples = inner.split('),(')
            if len(tuples) == 3:
                def _parse_el(s):
                    s = s.strip()
                    try:
                        return int(s)
                    except ValueError:
                        return s.strip("'\"")
                t1 = tuple(_parse_el(x) for x in tuples[0].strip('()').split(','))
                t2 = tuple(_parse_el(x) for x in tuples[1].strip('()').split(','))
                k = int(tuples[2].strip('()').split(',')[0])
                return ((t1, t2, k), l)

    def __get_var_name(self, root, e, k):
        n0 = tuple(int(x) for x in e[0])
        n1 = tuple(int(x) for x in e[1])
        if len(e) < 3 or e[2] == -1:
            i, g1, t1, q1 = e[0]
            j, g2, t2, q2 = e[1]
            matching_edges = [f for f in self.Ntl.Ntl.edges if e[0] == f[0] and e[1] == f[1] and self.Ntl.edge_charges[f] == max(0, g2 - g1) and self.Ntl.edge_dists[f] == max(g1 - g2, 0)]
            if len(matching_edges) == 0:
                return root + "[(" + str(n0) + "," + str(n1) + ",(0)), " + str(k) + "]"

            return root + "[(" + str(n0) + "," + str(n1) + ",(" + str(int(matching_edges[0][2])) + ")), " + str(k) + "]"

        else:
            return root + "[(" + str(n0) + "," + str(n1) + ",(" + str(int(e[2])) + ")), " + str(k) + "]"
    
    def multi_in_edges(self, v):
        return list(self.Ntl.Ntl.in_edges(v, keys=True))

    def multi_out_edges(self, v):
        return list(self.Ntl.Ntl.out_edges(v, keys=True))

    def __constr_flow_bal_load(self, v):
        for k in range(self.K):
            if (v[0] not in self.param['sources'] or v[2] != self.param['min_time']) and (v[0] not in self.param['sinks'] or v[2] != self.param['T'] - 1):
                self._cons[('flow_bal_load', v, k)] = self.model.addConstr(gp.quicksum(self.x_load[e, k] for e in self.multi_in_edges(v)) == gp.quicksum(self.x_load[e, k] for e in self.multi_out_edges(v)), name='flow_bal_load[' + str(v) + ',' + str(k) + ']')


    def __constr_flow_bal_ener(self, v):
        for k in range( self.K):
            if (v[2] != self.param['min_time']) and ( v[2] != self.param['T'] - 1):
                self._cons[('flow_bal_ener', v, k)] = self.model.addConstr(gp.quicksum(self.x_ener[e, k] for e in self.multi_in_edges(v)) == gp.quicksum(self.x_ener[e, k] for e in  self.multi_out_edges(v)), name='flow_bal_ener[' + str(v) + ',' + str(k) + ']')


    def __constr_init_cond_x(self, i):
        LHS = gp.quicksum(self.x_ener[e, k] for k in range( self.K) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, self.param['min_time'], self.Ntl.get_q(i,g))))
        RHS = self.n[i] + sum(v for nodes, v in self.param['D'].items() if nodes[0] == i)
        self._cons[('init_cond', i)] = self.model.addConstr(LHS == RHS, name='init_cond_x[' + str(i) +']')
        

    def __constr_end_cond_x(self, i):
        # constraint 1
        LHS = gp.quicksum(self.x_ener[e, k]  for k in range(self.K) for g in self.Ntl.charges[i] for e in self.multi_in_edges((i, g, self.param['T'] - 1, self.Ntl.get_q(i,g))))
        RHS = self.n[i] + sum(v for nodes, v in self.param['D'].items() if nodes[1] == i)
        self._cons[('end_cond_1', i)] = self.model.addConstr(LHS == RHS, name='end_cond_x[' + str(i) + ']_1')

        # constraint 2 on surplus tractors only
        if i in self.param['tractor_nodes']:
            min_charge = min([self.param['L']] + [self.N[i][j]['dL'] for j in self.N.neighbors(i) if j in self.param['charge_nodes']])
            LHS = gp.quicksum(self.x_ener[e, k]  for k in range(self.K) for g in self.Ntl.charges[i] for e in self.multi_in_edges((i, g, self.param['T'] - 1, self.Ntl.get_q(i,g))) if g >= min_charge)
            RHS = self.n[i] - self.param['N_tractors'] * self.z[i]
            self._cons[('end_cond_2', i)] = self.model.addConstr(LHS >= RHS, name='end_cond_x[' + str(i) + ']_2')
        


    def __constr_chargecap(self, i, t):
        t_index = self.Ntl.times[i].index(t)
        if t < self.param['T'] - 1:
            Delta = self.Ntl.times[i][t_index + 1] - self.Ntl.times[i][t_index]
        else:
            Delta = 1
        if i in self.param['charge_nodes']:
            # an existing station keeps its current capacity for free: a_c[i] + 1 units
            units = self.a_c[i] + (1 if i in self._existing_stations else 0)
            self._cons[('charge_cap', i, t)] = self.model.addConstr(gp.quicksum(self.Ntl.edge_charges[e] * (self.x_ener[e, k]) for k in range(self.K) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, t, self.Ntl.get_q(i,g))) if self.Ntl.edge_types[e] == 'charge')   <= Delta * self.param["charge_rate"][i] * units, name='charge_cap[' + str(i) + ',' + str(t) + ']')
        else:
            self._cons[('charge_cap', i, t)] = self.model.addConstr(gp.quicksum(self.Ntl.edge_charges[e] * (self.x_ener[e, k]) for k in range(self.K) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, t, self.Ntl.get_q(i,g))) if self.Ntl.edge_types[e] == 'charge') == 0, name='charge_cap[' + str(i) + ',' + str(t) + ']')

    

    def __constr_edge(self, e):
        if self.Ntl.edge_types[e] == 'transit_L': # or self.Ntl.edge_types[e] == 'charge':
            for k in range(self.K):
                self._cons[('edge', e, k)] = self.model.addConstr(self.x_load[e, k] == 0, name='e_load_constr[' + str(e) + ',' + str(k) + ']')
        elif self.Ntl.edge_types[e] == 'transit_H':
            i, g, t, q = e[1]
            if i in self.param['battery_nodes']:
                for k in range(self.K):
                    self._cons[('edge', e, k)] = self.model.addConstr(self.x_load[e, k] == self.x_ener[e, k], name='e_load_ener_bat[' + str(e) + ',' + str(k) + ']')
            else: 
                for k in range(self.K):
                    self._cons[('edge', e, k)] = self.model.addConstr(self.x_ener[e, k] >= self.x_load[e, k], name='e_load_ener[' + str(e) + ',' + str(k) + ']')
        elif self.Ntl.edge_types[e] == 'swap':
            for k in range(self.K):
                self._cons[('edge', e, k)] = self.model.addConstr(self.x_ener[e, k] == 0, name='e_swap_x[' + str(e) + ',' + str(k) + ']')

    def __constr_demand(self):
        assert 'D' in self.param.keys()
        if 'D' in self.param.keys():
            for k, (nodes, v) in enumerate(self.param['D'].items()):
                src, sink = nodes
                self._cons[('demand_src', k)] = self.model.addConstr(gp.quicksum(self.x_load[e, k] for e in self.Ntl.Ntl.edges if e[0][0] == src and e[0][2] == self.param['min_time']) - gp.quicksum(self.x_load[e, k] for e in self.Ntl.Ntl.edges if e[1][0] == src and e[1][2] == 0) - gp.quicksum(self.x_load[e, k] for e in self.Ntl.Ntl.edges if e[1][0] == src and e[1][2] == self.param['T'] - 1)  == v,name=f"demand_{src}_{k}_src")
                self._cons[('demand_dest', k)] = self.model.addConstr(gp.quicksum(self.x_load[e, k] for e in self.Ntl.Ntl.edges if e[1][0] == sink and e[1][2] == self.param['T'] - 1) - gp.quicksum(self.x_load[e, k] for e in self.Ntl.Ntl.edges if e[0][0] == sink and e[0][2] == self.param['min_time'])  == v,name=f"demand_{sink}_{k}_dest")
        # else:
        #     for i in self.param['sources']:
        #         self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[0][0] == i and e[0][2] == self.param['min_time']) == self.D[i], name=f"demand_{i}_src")
        #     for i in self.param['sinks']:
        #         self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[1][0] == i and e[1][2] == self.param['T'] - 1) == self.D[i], name=f"demand_{i}_dest")

    def construct_model(self, penalty=False):
        self.model = gp.Model()
        self.model.setParam('MIPGap', 0.05)
        # Constraint handles by key (node, edge, commodity, ...). update_model removes
        # constraints through these rather than by name: names embed str(node), which
        # differs between int and np.int64 values of the same node.
        self._cons = dict()
        self.x_load = dict()
        self.x_ener = dict()
        #self.D = dict()
        self.K = 1
        if 'D' in self.param.keys():
            self.K = len(self.param['D'])
        # origin of each commodity, in the order of param['D'] (as in __constr_demand)
        self._origins = [nodes[0] for nodes in self.param['D'].keys()] if 'D' in self.param else []
        # for i in self.param['sources']:
        #     self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
        # for i in self.param['sinks']:
        #     self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
 
        # create variables
        for e in self.Ntl.Ntl.edges:
            for k in range(self.K):
                self.x_load[e, k] = self.model.addVar(lb=0, ub=self._load_ub(e, k), vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_load', e, k))
                self.x_ener[e, k] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_ener', e, k))
        
           
        self.n = dict()
        self.a_c = dict()   # number of chargers installed at i
        self.a_s = dict()   # 1 if a swap site is installed at i (battery or tractor nodes)
        self.a = self.a_c   # older name, kept for code that reads inst.a
        self.z = dict()
        self._existing_stations = set(self.param.get('existing_charge_nodes', []))
        for i in self.N.nodes:
            if i in self.param['tractor_nodes'] or i in self.param['battery_nodes']:
                self.n[i] = self.model.addVar(lb = 0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
                if i in self.param['tractor_nodes']: 
                    self.z[i] = self.model.addVar(lb = 0, vtype=gp.GRB.BINARY, name="z[" + str(i) + "]")
            else:
                self.n[i] = self.model.addVar(lb = 0, ub=0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
                self.z[i] = self.model.addVar(lb = 0, ub=0, vtype=gp.GRB.BINARY, name="z[" + str(i) + "]")
            if i in self.param['charge_nodes']:
                self.a_c[i] = self.model.addVar(lb = 0, vtype=gp.GRB.INTEGER, name='a_c[' + str(i) + ']')
            else:
                self.a_c[i] = self.model.addVar(lb=0, ub=0,  vtype=gp.GRB.INTEGER, name='a_c[' + str(i) + ']')
            swap_node = i in self.param['tractor_nodes'] or i in self.param['battery_nodes']
            self.a_s[i] = self.model.addVar(lb=0, ub=1 if swap_node else 0, vtype=gp.GRB.BINARY, name='a_s[' + str(i) + ']')
        self.model.update()

        # surplus units only at an installed swap site: n[i] <= M * a_s[i], M = total demand
        M_site = sum(self.param['D'].values()) if 'D' in self.param else 0
        for i in self.N.nodes:
            if i in self.param['tractor_nodes'] or i in self.param['battery_nodes']:
                self._cons[('site_cap', i)] = self.model.addConstr(self.n[i] <= M_site * self.a_s[i],
                                                                   name='site_cap[' + str(i) + ']')
        
        for v in self.Ntl.Ntl.nodes:
            self.__constr_flow_bal_ener(v)
            self.__constr_flow_bal_load(v)
        for i in self.N.nodes:
            self.__constr_init_cond_x(i)
            self.__constr_end_cond_x(i)
            for t in self.Ntl.times[i]:
                self.__constr_chargecap(i, t)
 
        #self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['tractor_nodes']) <= self.param['N_tractors'], name='tractor_limit')
        #self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['battery_nodes']) <= self.param['N_batteries'], name='battery_limit')
        #self.model.addConstr(gp.quicksum(self.a[i] for i in self.param['charge_nodes'] if i not in self.param.get('existing_charge_nodes', [])) <= self.param['N_chargers'], name='charger_limit-1')
        #self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['charge_nodes']) <= 0, name='charger_limit-2')

        for e in self.Ntl.Ntl.edges:
            self.__constr_edge(e)
             
        self.__constr_demand()

        if penalty:
            raise NotImplementedError
        self.set_objective()
        self.model.update()

    def set_objective(self):
        obj_expr = (
            gp.quicksum(self.x_load[e, k] * self.Ntl.edge_times[e] * self.param['value_for_time']
                        for e in self.Ntl.Ntl.edges for k in range(self.K))
            + gp.quicksum(self.x_ener[e, k] * self.Ntl.edge_charges[e] * self.param['charge_cost'][(e[0][0], e[0][2])]
                          for e in self.Ntl.Ntl.edges for k in range(self.K) if self.Ntl.edge_charges[e] > 0)
            + gp.quicksum(self.param['stat_cost'][i] * self.a_c[i] for i in self.param['charge_nodes'])
            + gp.quicksum(self.param.get('site_install_cost', 0) * self.a_s[i]
                          for i in set(self.param['tractor_nodes']) | set(self.param['battery_nodes']))
            + gp.quicksum(self.param['tractor_cost'] * self.n[i] for i in self.param['tractor_nodes'])
            + gp.quicksum(self.param['battery_cost'] * self.n[i] for i in self.param['battery_nodes'])
            + gp.quicksum(self.param['battery_cost'] * self.z[i] for i in self.param['tractor_nodes'])
        )

        if 'lagrange_multipliers' in self.param.keys():
            obj_expr -= gp.quicksum(self.param['lagrange_multipliers']['load'].get((e[0], e[1]), 0) * self.x_load[e, k]
                                    for e in self.Ntl.Ntl.edges for k in range(self.K))
            obj_expr -= gp.quicksum(self.param['lagrange_multipliers']['ener'].get((e[0], e[1]), 0) * self.x_ener[e, k]
                                    for e in self.Ntl.Ntl.edges for k in range(self.K))

        if 'D' in self.param.keys():
            self.model.setObjective(obj_expr, gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(sum(self.D[i] for i in self.param['sinks']), gp.GRB.MAXIMIZE)

    UPDATE_POLICIES = ('smart_update', 'simple_update', 'simple_w_incoming_update', 'local_rebuild_update')

    def _set_update_policy(self, policy):
        """Use `policy` for later network updates (Instance and its network share the choice)."""
        self.config = [c for c in self.config if c not in self.UPDATE_POLICIES] + [policy]
        self.Ntl.config = [c for c in self.Ntl.config if c not in self.UPDATE_POLICIES] + [policy]

    def _apply_initial_discretization(self, source):
        """Start from an earlier run's discretization (see 'initial_discretization')."""
        if isinstance(source, dict):
            disc = {'times': source.get('times'), 'charges': source.get('charges'),
                    'iteration': None, 'elapsed': 0, 'path': None, 'start': {}}
        else:
            disc = discretization_from_log(source)
            start = disc['start']
            if start.get('T') is not None and start['T'] != self.param['T']:
                raise ValueError(f"{source}: logged run has T={start['T']}, this instance T={self.param['T']}")
            kinds = {'default', 'heuristic'}
            logged = kinds & set(start.get('config') or [])
            mine = kinds & set(self.config)
            if logged and logged != mine:
                raise ValueError(f"{source}: logged run used the {logged} network, this instance {mine}")
        unknown = {int(i) for grid in (disc['times'], disc['charges']) for i in (grid or {})} - set(self.N.nodes)
        if unknown:
            raise ValueError(f"initial discretization has stations not in this network: {sorted(unknown)[:10]}")
        added = self.Ntl.set_discretization(disc['times'], disc['charges'])
        if self.verbose:
            print(f"Resumed discretization from {disc['path'] or 'dict'} (iteration {disc['iteration']}): "
                  f"{added} points added")
        return {'path': disc['path'], 'iteration': disc['iteration'], 'points_added': added,
                'lower_bound': disc.get('lower_bound'), 'elapsed': disc.get('elapsed') or 0}

    def _load_ub(self, e, k):
        """
        Upper bound of x_load[e, k]: 0 if e leaves commodity k's origin after
        param['latest_origin_time'] (the load must have left its origin by then; this
        is the constraint sum_{e in out(o_k, g, t)} x^k_e = 0 for t > latest_origin_time,
        written as bounds since flows are non-negative), otherwise unbounded.
        """
        latest = self.param.get('latest_origin_time')
        if latest is not None and e[0][0] == self._origins[k] and e[0][2] > latest:
            return 0
        return gp.GRB.INFINITY

    def _remove_constr(self, key):
        c = self._cons.pop(key, None)
        if c is not None:
            self.model.remove(c)

    def update_model(self, penalty=False):
        """Incrementally update the Gurobi model to reflect changes in self.Ntl.Ntl.
        Called instead of construct_model on DDD iterations after the first.
        """
        current_edges = set(self.Ntl.Ntl.edges)
        model_edges   = set(e for e, k in self.x_load.keys())
        new_edges     = current_edges - model_edges
        removed_edges = model_edges   - current_edges

        # 1. Disable variables for removed edges (fix bounds to 0) and remove their edge constraints
        for e in removed_edges:
            for k in range(self.K):
                self.x_load[e, k].lb = self.x_load[e, k].ub = 0
                self.x_ener[e, k].lb = self.x_ener[e, k].ub = 0
                self._remove_constr(('edge', e, k))

        # 2. Add variables for new edges
        for e in new_edges:
            for k in range(self.K):
                self.x_load[e, k] = self.model.addVar(lb=0, ub=self._load_ub(e, k), vtype=gp.GRB.INTEGER,
                                                      name=self.__get_var_name('x_load', e, k))
                self.x_ener[e, k] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER,
                                                      name=self.__get_var_name('x_ener', e, k))
 

        self.model.update()

        # 3. Add edge constraints for new edges
        for e in new_edges:
            self.__constr_edge(e)

        # 4. Identify affected network nodes and original nodes
        changed_edges   = new_edges | removed_edges
        affected_vnodes = {e[0] for e in changed_edges} | {e[1] for e in changed_edges}
        affected_i      = {v[0] for v in affected_vnodes}

        # 5. Update flow balance constraints for affected network nodes
        for v in affected_vnodes:
            for k in range(self.K):
                self._remove_constr(('flow_bal_load', v, k))
                self._remove_constr(('flow_bal_ener', v, k))
        self.model.update()
        for v in affected_vnodes:
            self.__constr_flow_bal_load(v)
            self.__constr_flow_bal_ener(v)

        # 6. Update init/end condition constraints for affected original nodes
        for i in affected_i:
            for key in (('init_cond', i), ('end_cond_1', i), ('end_cond_2', i)):
                self._remove_constr(key)
        self.model.update()
        for i in affected_i:
            if i in self.N.nodes:
                self.__constr_init_cond_x(i)
                self.__constr_end_cond_x(i)

        # 7. Update charge cap constraints for affected (i, t) pairs.
        # Include all times for affected nodes in case new time steps were added.
        affected_it = {(e[0][0], e[0][2]) for e in changed_edges}
        for i in affected_i:
            for t in self.Ntl.times[i]:
                affected_it.add((i, t))
        for i, t in affected_it:
            self._remove_constr(('charge_cap', i, t))
        self.model.update()
        for i, t in affected_it:
            if i in self.N.nodes:
                self.__constr_chargecap(i, t)

        # 8. Update demand constraints (they sum over all edges at the sources and sinks)
        for k in range(self.K):
            self._remove_constr(('demand_src', k))
            self._remove_constr(('demand_dest', k))
        self.model.update()
        self.__constr_demand()

        # 9. Reset objective
        self.set_objective()

        self.model.update()

    


    """
    Functions for solving the instance
    
    """

    def run_DDD(self, LB=0, LP=True):
        # LP_soln, LP_val, solve_properties = self.solve(True, penalty=False, LB=LB)
        # print("OPTIMAL LP value", LP_val)
        # if LP:
        #     return LP_soln, LP_val, solve_properties
        # print(LP_soln)
        final_soln, final_val, final_solve_prop = self.solve(False)
        # for key, val in solve_properties.items():
        #     final_solve_prop[key + '_LP'] = val
        # final_solve_prop['LP_val'] = LP_val
        #final_solve_prop['total_iterations'] = final_solve_prop['n_iterations'] + solve_properties['n_iterations']
        return final_soln, final_val, final_solve_prop
        
    

    def resolve_infeasibility(self, LP_relax, verbose):
        if verbose and self.verbose:
            print("Infeasible... recomputing full network")
        self.Ntl.construct()
        self.construct_model()
        M = self.model.relax() if LP_relax else self.model
        M.optimize()

        if M.status != gp.GRB.OPTIMAL:
            # Use a time limit for IIS to avoid hanging on large models
            M.Params.IISMethod = 1
            M.computeIIS()
            M.write('my_iis.ilp')
            if self.verbose:
                print("Irreducible Inconsistent Subsystem (IIS):")
                for c in M.getConstrs():
                    if c.IISConstr:
                        print(f"Infeasible constraint: {c.ConstrName}")
            raise RuntimeError("Model is infeasible.")
        if M.status == gp.GRB.OPTIMAL and verbose and self.verbose:
            print("Solution found after full network recomputation!")
            
    
           
    @staticmethod
    def _objective_bound(M):
        """Proven lower bound of the solved relaxation.

        Each relaxation MIP is solved only to MIPGap, so ObjVal can exceed the
        relaxation optimum; ObjBound cannot. An LP has no ObjBound, but its
        optimal ObjVal is exact.
        """
        try:
            return M.ObjBound
        except (gp.GurobiError, AttributeError):
            return M.ObjVal

    def _log_ddd(self, record, mode='a'):
        """Append one JSON record to self.param['ddd_log'], if set.

        The file is reopened for every record so that everything up to the
        last finished iteration is on disk even if the process is killed.
        """
        path = self.param.get('ddd_log')
        if not path:
            return
        with open(path, mode) as f:
            f.write(json.dumps(_jsonable(record)) + '\n')

    def solve(self, LP_relax=False, penalty=True, verbose=True, LB=0):
        # A resumed run (see 'initial_discretization') continues the earlier run's counts:
        # iterations are numbered on from its last logged one and elapsed time includes its
        # running time, so both are totals over all attempts (MAX_ITER caps the total).
        resumed = self.resumed_from or {}
        n_iter = 1 + (resumed.get('iteration') or 0)
        curr_LB = 0
        iter_times = []
        iter_graph_sizes = []
        _interrupted = False

        # Convert SIGTERM (kill) into a KeyboardInterrupt so the existing
        # exception handling below also fires on external kill signals.
        def _sigterm_handler(signum, frame):
            raise KeyboardInterrupt(f"signal {signum} received")

        _prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)

        solve_start = time.time() - (resumed.get('elapsed') or 0)
        last_bound = None
        # construct() builds the DDD network of the initial discretization; only the older
        # update policies can leave it in a state that is not (see 'local_rebuild_update')
        network_is_ddd = True
        self._log_ddd({'event': 'start',
                       'started': datetime.datetime.now().isoformat(timespec='seconds'),
                       'config': self.config, 'LP_relax': LP_relax,
                       'T': self.param['T'], 'L': self.param['L'],
                       'step_size': self.param['step_size'], 'MAX_ITER': self.param['MAX_ITER'],
                       'latest_origin_time': self.param.get('latest_origin_time'),
                       'resumed_from': self.resumed_from,
                       # node coordinates, if the network has them (formerly printed every iteration)
                       'node_positions': {i: self.N.nodes[i]['pos'] for i in self.N.nodes
                                          if 'pos' in self.N.nodes[i]}},
                      mode='w')

        last_valid_bound = None
        # optional fast first phase (see 'fast_update'): the configured policy is restored,
        # with a full rebuild, once the elapsed time reaches fast['seconds']
        fast = self.param.get('fast_update')
        final_policy = next((c for c in self.config if c in self.UPDATE_POLICIES), None)
        fast_active = bool(fast) and time.time() - solve_start < fast['seconds']
        if fast_active:
            self._set_update_policy(fast['policy'])
        try:
            if self.verbose:
                print("'''' LOGGING ''''")
            while n_iter < self.param['MAX_ITER']:
                try:
                    iter_start = time.time()

                    if fast_active and time.time() - solve_start >= fast['seconds']:
                        # end of the fast phase: make the network the DDD network of the
                        # current discretization again, then refine with the configured policy
                        fast_active = False
                        self._set_update_policy(final_policy)
                        self.Ntl.construct()
                        self.construct_model()
                        network_is_ddd = True
                        self._log_ddd({'event': 'policy_switch', 'iteration': n_iter,
                                       'from': fast['policy'], 'to': final_policy,
                                       'elapsed': time.time() - solve_start,
                                       'num_edges': len(self.Ntl.Ntl.edges)})

                    if LP_relax:
                        M = self.model.relax()
                    else:
                        M = self.model

                    M.optimize()

                    if M.status != gp.GRB.OPTIMAL:
                        self.resolve_infeasibility(LP_relax, verbose)
                        continue
                    
                    if self.verbose:
                        print(f"Iteration {n_iter}: obj {M.ObjVal}, network {len(self.Ntl.Ntl.nodes)} nodes, {len(self.Ntl.Ntl.edges)} edges")

                    x_load = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_load" in v.VarName}
                    x_ener = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_ener" in v.VarName}
                    a_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if v.VarName.startswith("a_c[")}
                    as_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if v.VarName.startswith("a_s[")}
                    n_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "n[" in v.VarName}
                    z_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "z[" in v.VarName}
                    curr_LB = M.ObjVal
                    # The bound is a lower bound on the true optimum only if this network is
                    # a relaxation, i.e. the DDD network (always, except under the fast policy).
                    relaxation_bound = self._objective_bound(M)
                    if network_is_ddd:
                        last_valid_bound = relaxation_bound
                    last_bound = last_valid_bound
                    # Discretization and size of the network this relaxation was solved on
                    # (before update() refines it for the next iteration).
                    self._log_ddd({'event': 'iteration', 'iteration': n_iter,
                                   'lower_bound': relaxation_bound if network_is_ddd else None,
                                   'relaxation_bound': relaxation_bound,
                                   'update_policy': fast['policy'] if fast_active else final_policy,
                                   'objective': M.ObjVal,
                                   'num_edges': len(self.Ntl.Ntl.edges),
                                   'num_nodes': len(self.Ntl.Ntl.nodes),
                                   'elapsed': time.time() - solve_start,
                                   'times': self.Ntl.times, 'charges': self.Ntl.charges})

                    assert curr_LB >= LB, "objective value too small"
                   
                    if self.verbose:
                        print('a_c:', dict((k, v) for k, v in a_.items() if v > 0))
                        print('a_s:', dict((k, v) for k, v in as_.items() if v > 0))
                        print('n:', dict((k, v) for k, v in n_.items() if v > 0))

                    x_loads = dict()
                    x_eners = dict()
                    for k in range(self.K):
                        x_loads[k] = {key[0]: flow for key, flow in x_load.items() if key[1] == k and flow > 1e-6}
                        x_eners[k] = {key[0]: flow for key, flow in x_ener.items() if key[1] == k and flow > 1e-6}
                        if self.verbose:
                            print(f'x_load_{k}:', x_loads[k])
                            print(f'x_ener_{k}:', x_eners[k])

                    iter_graph_sizes.append((len(self.Ntl.Ntl.nodes), len(self.Ntl.Ntl.edges)))

                    # Termination (Algorithm 1): the solution extends to a feasible solution of
                    # the full charge-time network if every edge carrying load or energy flow is
                    # exact, since exact edges are edges of the full network with the same costs.
                    # It is then optimal, as its cost is a lower bound -- provided the network is
                    # a relaxation, which is only guaranteed when it is the true DDD network.
                    artificial = sorted({e for k in range(self.K) for e in list(x_loads[k]) + list(x_eners[k])
                                         if not self.Ntl.is_exact(e)}, key=str)

                    if not artificial:
                        if network_is_ddd:
                            if verbose and self.verbose:
                                print("feasible solution found!")
                                print("'''''''''''''''''")
                            avg_time = sum(iter_times) / len(iter_times) if iter_times else 0
                            avg_nodes = sum(s[0] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
                            avg_edges = sum(s[1] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
                            # every edge used is exact, so the network flow is the real solution
                            flows = {name: {k: {} for k in range(self.K)} for name in ('x_load', 'x_ener')}
                            for name, per_k in (('x_load', x_loads), ('x_ener', x_eners)):
                                for k in range(self.K):
                                    for e, flow in per_k[k].items():
                                        flows[name][k][(e[0], e[1])] = flows[name][k].get((e[0], e[1]), 0) + flow
                            # 'a' kept as the older name of a_c
                            soln = {'x_load': flows['x_load'], 'x_ener': flows['x_ener'], 'a_c': a_, 'a_s': as_,
                                    'a': a_, 'n': n_}
                            self._log_ddd({'event': 'final', 'status': 'converged', 'iteration': n_iter,
                                           'objective': M.ObjVal, 'lower_bound': last_bound,
                                           'num_edges': len(self.Ntl.Ntl.edges),
                                           'elapsed': time.time() - solve_start,
                                           'times': self.Ntl.times, 'charges': self.Ntl.charges,
                                           'solution': dict(soln, z=z_)})
                            return soln, M.ObjVal, {'n_iterations': n_iter, 'size_of_graph': (len(self.Ntl.Ntl.nodes), len(self.Ntl.Ntl.edges)), 'avg_time_per_iter': avg_time, 'avg_graph_size_per_iter': (avg_nodes, avg_edges)}
                        # The older update policies can leave a network that is not a relaxation,
                        # so this solution may not be optimal: rebuild the DDD network for the
                        # current discretization, re-solve, and test the new solution.
                        if self.verbose:
                            print("Feasible solution on a patched network; rebuilding the DDD network and re-solving")
                        self.Ntl.construct()
                        self.construct_model()
                        network_is_ddd = True
                        n_iter += 1
                        iter_times.append(time.time() - iter_start)
                        continue

                    refined = False
                    updates = self.Ntl.update_all([[x_loads[k], x_eners[k]] for k in range(self.K)])
                    for new_edge, removed_edge in updates:
                        refined = refined or new_edge is not None
                        self.penalties[removed_edge] = PEN_CONST if penalty else 0

                        if verbose and self.verbose:
                            print("New edge added: ", new_edge, " Edge removed: ", removed_edge)

                    if not refined:
                        # The update policy found nothing to refine (e.g. a path that runs out of
                        # battery is cut short and every edge of the shortened path exists), but
                        # the solution still uses artificial edges: correct those directly.
                        self.Ntl.refine_artificial_edges(artificial)
                    network_is_ddd = 'local_rebuild_update' in self.config
                    self.update_model(penalty=penalty)

                    if self.verbose:
                        print("Solution cannot be extended to a feasible solution, updating network...")
                        print(f"{len(artificial)} artificial edges carry flow")
                        for i in a_.keys():
                            if a_[i] > 0 and self.verbose:
                                print("a_c[", i, "] = ", a_[i])
                        for i in n_.keys():
                            if n_[i] > 0 and self.verbose:
                                print("n[", i, "] = ", n_[i])
                        for i in z_.keys():
                            if z_[i] > 0 and self.verbose:
                                print("z[", i, "] = ", z_[i])
                        if self.verbose:
                            print("'''''''''''''''''")
                    n_iter += 1
                    iter_times.append(time.time() - iter_start)
                except KeyboardInterrupt:
                    _interrupted = True
                    break
        finally:
            signal.signal(signal.SIGTERM, _prev_sigterm)
            if _interrupted and self.param.get('snapshot_on_interrupt'):
                self.dump_to_json()

        self._log_ddd({'event': 'final', 'status': 'interrupted' if _interrupted else 'max_iter',
                       'iteration': n_iter, 'objective': curr_LB, 'lower_bound': last_bound,
                       'num_edges': len(self.Ntl.Ntl.edges),
                       'elapsed': time.time() - solve_start, 'solution': None})
        avg_time = sum(iter_times) / len(iter_times) if iter_times else 0
        avg_nodes = sum(s[0] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
        avg_edges = sum(s[1] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
        return {'x_load': {}, 'x_ener':{}, 'a': {}, 'n':{}}, curr_LB, {'n_iterations': n_iter, 'timeout':True, 'size_of_graph': (len(self.Ntl.Ntl.nodes), len(self.Ntl.Ntl.edges)), 'avg_time_per_iter': avg_time, 'avg_graph_size_per_iter': (avg_nodes, avg_edges)}


    """
    Functions for testing and debugging
    """

    def dump_to_json(self, filepath=None):
        """Serialize the current instance state (params + Ntl graph) to a JSON file.

        Tuple keys and node IDs are stored as their string representations so the
        file is valid JSON.  The file can be used to reconstruct the network state
        for debugging or warm-starting after an interrupted solve.

        Parameters
        ----------
        filepath : str, optional
            Destination path.  Defaults to ``instance_snapshot_<timestamp>.json``
            in the current directory.

        Returns
        -------
        str
            The path of the written file.
        """
        if filepath is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'instance_snapshot_{ts}.json'

        graph_data = {
            'nodes': [str(n) for n in self.Ntl.Ntl.nodes()],
            'edges': [
                {'u': str(u), 'v': str(v), 'key': k}
                for u, v, k in self.Ntl.Ntl.edges(keys=True)
            ],
            'edge_types':   {str(e): _jsonable(v) for e, v in self.Ntl.edge_types.items()},
            'edge_times':   {str(e): _jsonable(v) for e, v in self.Ntl.edge_times.items()},
            'edge_dists':   {str(e): _jsonable(v) for e, v in self.Ntl.edge_dists.items()},
            'edge_charges': {str(e): _jsonable(v) for e, v in self.Ntl.edge_charges.items()},
            'charges':      _jsonable(self.Ntl.charges),
            'times':        _jsonable(self.Ntl.times),
        }

        snapshot = {
            'param': _jsonable(self.param),
            'graph': graph_data,
        }

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        if self.verbose:
            print(f"Instance snapshot written to {filepath}")
        return filepath

    def diff_models(self, old_model):
        equal = True 
        old_vars = set(v.VarName for v in old_model.getVars())
        new_vars = set(v.VarName for v in self.model.getVars())
        added_vars = new_vars - old_vars
        removed_vars = old_vars - new_vars
        if len(added_vars) > 0:
            equal = False
            l = min(len(added_vars), 10)
            if self.verbose:
                print("New variables added: ", list(added_vars)[:l])
        if len(removed_vars) > 0:
            equal = False
            l = min(len(removed_vars), 10)
            if self.verbose:
                print("Variables removed: ", list(removed_vars)[:l])
        if not equal:
            return False 

        old_constrs = set(c.ConstrName for c in old_model.getConstrs())
        new_constrs = set(c.ConstrName for c in self.model.getConstrs())
        added_constrs = new_constrs - old_constrs
        removed_constrs = old_constrs - new_constrs
        if len(added_constrs) > 0:
            equal = False
            if self.verbose:
                print("New constraints added: ", added_constrs)
        if len(removed_constrs) > 0:
            equal = False
            if self.verbose:
                print("Constraints removed: ", removed_constrs)

        return equal
