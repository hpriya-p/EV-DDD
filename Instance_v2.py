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
        #   stat_cost     (dict)  – {node: cost}; fixed cost per charger installed at node
        #   surplus_cost  (dict)  – {node: cost}; cost per surplus truck stationed at node
        #   MAX_ITER      (int)   – maximum number of column-generation / update iterations
        #   lagrange_multipliers (dict) – (optional) {'load': {edge: mu}, 'ener': {edge: mu}};
        #                                 Lagrange multipliers subtracted from the objective
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
        if self.verbose:
            print("Size of network: ", len(self.Ntl.Ntl.nodes), " nodes, ", len(self.Ntl.Ntl.edges), " edges")
        if seed is not None:
            self.seed(seed)
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
            self._cons[('charge_cap', i, t)] = self.model.addConstr(gp.quicksum(self.Ntl.edge_charges[e] * (self.x_ener[e, k]) for k in range(self.K) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, t, self.Ntl.get_q(i,g))) if self.Ntl.edge_types[e] == 'charge')   <= Delta * self.param["charge_rate"][i] * self.a[i], name='charge_cap[' + str(i) + ',' + str(t) + ']')
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
        # for i in self.param['sources']:
        #     self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
        # for i in self.param['sinks']:
        #     self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
 
        # create variables
        for e in self.Ntl.Ntl.edges:
            for k in range(self.K):
                self.x_load[e, k] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_load', e, k))
                self.x_ener[e, k] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_ener', e, k))
        
           
        self.n = dict()
        self.a= dict()
        self.z = dict()
        for i in self.N.nodes:
            if i in self.param['tractor_nodes'] or i in self.param['battery_nodes']:
                self.n[i] = self.model.addVar(lb = 0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
                if i in self.param['tractor_nodes']: 
                    self.z[i] = self.model.addVar(lb = 0, vtype=gp.GRB.BINARY, name="z[" + str(i) + "]")
            else:
                self.n[i] = self.model.addVar(lb = 0, ub=0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
                self.z[i] = self.model.addVar(lb = 0, ub=0, vtype=gp.GRB.BINARY, name="z[" + str(i) + "]")
            if i in self.param['charge_nodes']:
                self.a[i] = self.model.addVar(lb = 0, vtype=gp.GRB.INTEGER, name='a[' + str(i) + ']')
            else:
                self.a[i] = self.model.addVar(lb=0, ub=0,  vtype=gp.GRB.INTEGER, name='a[' + str(i) + ']')
        self.model.update()
        
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
            + gp.quicksum(self.param['stat_cost'][i] * self.a[i] for i in self.param['charge_nodes'])
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
                self.x_load[e, k] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER,
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
        n_iter = 1
        curr_LB = 0
        iter_times = []
        iter_graph_sizes = []
        _interrupted = False

        # Convert SIGTERM (kill) into a KeyboardInterrupt so the existing
        # exception handling below also fires on external kill signals.
        def _sigterm_handler(signum, frame):
            raise KeyboardInterrupt(f"signal {signum} received")

        _prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)

        solve_start = time.time()
        last_bound = None
        # construct() builds the DDD network of the initial discretization; only the older
        # update policies can leave it in a state that is not (see 'local_rebuild_update')
        network_is_ddd = True
        self._log_ddd({'event': 'start',
                       'started': datetime.datetime.now().isoformat(timespec='seconds'),
                       'config': self.config, 'LP_relax': LP_relax,
                       'T': self.param['T'], 'L': self.param['L'],
                       'step_size': self.param['step_size'], 'MAX_ITER': self.param['MAX_ITER']},
                      mode='w')

        try:
            if self.verbose:
                print("'''' LOGGING ''''")
            while n_iter < self.param['MAX_ITER']:
                try:
                    iter_start = time.time()

                    if LP_relax:
                        M = self.model.relax()
                    else:
                        M = self.model

                    M.optimize()

                    if M.status != gp.GRB.OPTIMAL:
                        self.resolve_infeasibility(LP_relax, verbose)
                        continue
                    
                    print("Iteration: ", n_iter)
                    print("Size of network: ", len(self.Ntl.Ntl.nodes), " nodes, ", len(self.Ntl.Ntl.edges), " edges")
                    print("Obj Value: ", M.ObjVal)

                    x_load = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_load" in v.VarName}
                    x_ener = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_ener" in v.VarName}
                    a_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "a[" in v.VarName}
                    n_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "n[" in v.VarName}
                    z_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "z[" in v.VarName}
                    curr_LB = M.ObjVal
                    last_bound = self._objective_bound(M)
                    # Discretization and size of the network this relaxation was solved on
                    # (before update() refines it for the next iteration).
                    self._log_ddd({'event': 'iteration', 'iteration': n_iter,
                                   'lower_bound': last_bound, 'objective': M.ObjVal,
                                   'num_edges': len(self.Ntl.Ntl.edges),
                                   'num_nodes': len(self.Ntl.Ntl.nodes),
                                   'elapsed': time.time() - solve_start,
                                   'times': self.Ntl.times, 'charges': self.Ntl.charges})

                    assert curr_LB >= LB, "objective value too small"
                   
                    if self.verbose:
                        print('a:', dict((k, v) for k, v in a_.items() if v > 0))
                        print('n:', dict((k, v) for k, v in n_.items() if v > 0))
                        try:
                            print({i: self.N.nodes[i]['pos'] for i in self.N.nodes})
                        except:
                            pass

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
                            soln = {'x_load': flows['x_load'], 'x_ener': flows['x_ener'], 'a': a_, 'n': n_}
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
                    for k in range(self.K):
                        new_edge, removed_edge = self.Ntl.update([x_loads[k], x_eners[k]])
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
                                print("a[", i, "] = ", a_[i])
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
            if _interrupted:
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
    
    def seed(self, soln):
        #TODO this function is faulty
        
        """
        Pre-populate self.Ntl.times and self.Ntl.charges from a prior solution's
        x_load flow dict, then completely rebuild the network and Gurobi model.

        For every edge with positive flow in x_load, the exact (t, g) values
        used by each endpoint are inserted into self.Ntl.times and self.Ntl.charges.
        The charge-time network is then cleared and reconstructed from scratch,
        followed by a full rebuild of the Gurobi model.

        Parameters
        ----------
        x_load : dict
            Edge-key -> flow value mapping as returned by run_DDD()[0]['x_load'].
        """
        # 1. Insert breakpoints from the prior solution
        modified = False
        for edge_dict in soln['x_load'].values():
            for e in edge_dict.keys():
                (i, g1, t1, _q1), (j, g2, t2, _q2) = e[0], e[1]
                if g1 not in self.Ntl.charges[i]:
                    self.Ntl.charges[i].append(g1)
                    self.Ntl.charges[i].sort()
                    modified = True
                if t1 not in self.Ntl.times[i]:
                    self.Ntl.times[i].append(t1)
                    self.Ntl.times[i].sort()
                    modified = True
                if g2 not in self.Ntl.charges[j]:
                    self.Ntl.charges[j].append(g2)
                    self.Ntl.charges[j].sort()
                    modified = True
                if t2 not in self.Ntl.times[j]:
                    self.Ntl.times[j].append(t2)
                    self.Ntl.times[j].sort()
                    modified = True

        modified = False
        for edge_dict in soln['x_ener'].values():
            for e in edge_dict.keys():
                (i, g1, t1, _q1), (j, g2, t2, _q2) = e[0], e[1]
                if g1 not in self.Ntl.charges[i]:
                    self.Ntl.charges[i].append(g1)
                    self.Ntl.charges[i].sort()
                    modified = True
                if t1 not in self.Ntl.times[i] and t1 < self.param['T']:
                    self.Ntl.times[i].append(t1)
                    self.Ntl.times[i].sort()
                    modified = True
                if g2 not in self.Ntl.charges[j]:
                    self.Ntl.charges[j].append(g2)
                    self.Ntl.charges[j].sort()
                    modified = True
                if t2 not in self.Ntl.times[j] and t2 < self.param['T']:
                    self.Ntl.times[j].append(t2)
                    self.Ntl.times[j].sort()
                    modified = True
        
        if modified: 
            for i in self.N.nodes:
                if self.verbose:
                    print(i)
                    print(self.Ntl.charges[i])
                    print(self.Ntl.times[i])
            
        self.Ntl.construct()
        self.construct_model()

