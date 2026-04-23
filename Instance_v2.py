from ChargeTimeNetwork_v2 import *
from copy import deepcopy
import numpy as np
import gurobipy as gp
import time
import json
import signal
import datetime

PEN_CONST = 10**6

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
        #   D             (dict)  – {node: (supply, demand)}; number of trucks supplied by node and demanded at node. 
        #   N_tractors    (int)   – maximum number of tractors that can be placed at tractor nodes
        #   N_batteries   (int)   – maximum number of batteries that can be placed at battery nodes
        #   N_chargers    (int)   – maximum number of chargers that can be installed at charge nodes
        #   charge_cost   (dict)  – {(node, time): cost}; per-unit charge cost at node i at time t
        #   stat_cost     (dict)  – {node: cost}; fixed cost per charger installed at node
        #   surplus_cost  (dict)  – {node: cost}; cost per surplus truck stationed at node
        #   MAX_ITER      (int)   – maximum number of column-generation / update iterations
        #   lagrange_multipliers (dict) – (optional) {'load': {edge: mu}, 'ener': {edge: mu}};
        #                                 Lagrange multipliers subtracted from the objective
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
        if len(set(self.config) & set(['smart_update', 'simple_update', 'simple_w_incoming_update'])) == 0:
            self.config = [x for x in self.config] + ['simple_w_incoming_update']

        # init charge-time-augmented network
        self.Ntl = ChargeTimeNetwork(N, self.param, config)
        print("Size of network: ", len(self.Ntl.Ntl.nodes), " nodes, ", len(self.Ntl.Ntl.edges), " edges")
        if seed is not None:
            self.seed(seed)
        if not network_only:
            self.construct_model()


        

    """
    Functions for model construction / update
    """
    
    def parse_var_name(self, var_name):
        # Assumes format: x_load[((i, g, t), (j, g2, t2), k)]
        inside = var_name[var_name.find('[')+1:var_name.find(']')]
        num_commas = inside.count(',')
        if num_commas == 0:
            i = int(inside.strip('()'))
            return i
        else:
            inside = inside.strip('[]')
            tuples = inside.split('),(')
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
                return (t1, t2, k)

    def __get_var_name(self, root, e):
        if len(e) < 3 or e[2] == -1:
            i, g1, t1, q1 = e[0]
            j, g2, t2, q2 = e[1]
            matching_edges = [f for f in self.Ntl.Ntl.edges if e[0] == f[0] and e[1] == f[1] and self.Ntl.edge_charges[f] == max(0, g2 - g1) and self.Ntl.edge_dists[f] == max(g1 - g2, 0)]
            if len(matching_edges) == 0:
                return root + "[" + str(e[0]) + "," + str(e[1]) +  ",(" + str(0) + ")]"
                
            return root + "[" + str(e[0]) + "," + str(e[1]) +  ",(" + str((matching_edges[0][2])) + ")]"

        else:
            return root + "[" + str(e[0]) + "," + str(e[1]) +  ",(" + str((e[2])) + ")]"
    
    def multi_in_edges(self, v):
        return list(self.Ntl.Ntl.in_edges(v, keys=True))

    def multi_out_edges(self, v):
        return list(self.Ntl.Ntl.out_edges(v, keys=True))

    def __constr_flow_bal_load(self, v):
        if (v[0] not in self.param['sources'] or v[2] != self.param['min_time']) and (v[0] not in self.param['sinks'] or v[2] != self.param['T'] - 1):
            self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.multi_in_edges(v)) == gp.quicksum(self.x_load[e] for e in self.multi_out_edges(v)), name='flow_bal_load[' + str(v) + ']')


    def __constr_flow_bal_ener(self, v):
        if (v[2] != self.param['min_time']) and ( v[2] != self.param['T'] - 1):
            self.model.addConstr(gp.quicksum(self.x_ener[e] for e in self.multi_in_edges(v)) == gp.quicksum(self.x_ener[e] for e in  self.multi_out_edges(v)), name='flow_bal_ener[' + str(v) + ']')


    def __constr_init_cond_x(self, i):
        if i not in self.param['sources']:
            self.model.addConstr(gp.quicksum(self.x_ener[e] for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, self.param['min_time'], self.Ntl.get_q(i,g)))) == self.n[i], name='init_cond_x[' + str(i) + ']')
        else:
            expr = 0
            for g in self.Ntl.charges[i]:
                for e in self.multi_out_edges((i, g, self.param['min_time'], self.Ntl.get_q(i,g))):
                    expr += self.x_ener[e]
            self.model.addConstr(expr== self.n[i] + self.D[i], name='init_cond_x[' + str(i) + ']')


    def __constr_end_cond_x(self, i):
        if i not in self.param['sinks']:
            self.model.addConstr(gp.quicksum(self.x_ener[e]  for g in self.Ntl.charges[i] for e in self.multi_in_edges((i, g, self.param['T'] - 1, self.Ntl.get_q(i,g)))) == self.n[i], name='end_cond_x[' + str(i) + ']')
        else:
            self.model.addConstr(gp.quicksum(self.x_ener[e]  for g in self.Ntl.charges[i] for e in self.multi_in_edges((i, g, self.param['T'] - 1, self.Ntl.get_q(i,g)))) == self.n[i] + self.D[i], name='end_cond_x[' + str(i) + ']')

    
    def __constr_chargecap(self, i, t):
        t_index = self.Ntl.times[i].index(t)
        if t < self.param['T'] - 1:
            Delta = self.Ntl.times[i][t_index + 1] - self.Ntl.times[i][t_index]
        else:
            Delta = 1
        if i in self.param['charge_nodes']:
            self.model.addConstr(gp.quicksum(self.Ntl.edge_charges[e] * (self.x_ener[e]) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, t, self.Ntl.get_q(i,g))) if self.Ntl.edge_types[e] == 'charge')   <= Delta * self.param["charge_rate"][i] * self.a[i], name='charge_cap[' + str(i) + ',' + str(t) + ']')
        else:
            self.model.addConstr(gp.quicksum(self.Ntl.edge_charges[e] * (self.x_ener[e]) for g in self.Ntl.charges[i] for e in self.multi_out_edges((i, g, t, self.Ntl.get_q(i,g))) if self.Ntl.edge_types[e] == 'charge') == 0, name='charge_cap[' + str(i) + ',' + str(t) + ']')

    

    def __constr_edge(self, e):
        if self.Ntl.edge_types[e] == 'transit_L': # or self.Ntl.edge_types[e] == 'charge':
            self.model.addConstr(self.x_load[e] == 0, name='e_load_constr[' + str(e) + ']')
        elif self.Ntl.edge_types[e] == 'transit_H':
            i, g, t, q = e[1]
            if i in self.param['battery_nodes']:
                self.model.addConstr(self.x_load[e] == self.x_ener[e], name='e_load_ener_bat[' + str(e) + ']')
            else: 
                self.model.addConstr(self.x_ener[e] >= self.x_load[e], name='e_load_ener[' + str(e) + ']')
        elif self.Ntl.edge_types[e] == 'swap':
            self.model.addConstr(self.x_ener[e] == 0, name='e_swap_x[' + str(e) + ']')

    def __constr_demand(self):
        if 'D' in self.param.keys():
            for k, v in self.param['D'].items():
                v_src, v_dest = v          
                self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[0][0] == k and e[0][2] == self.param['min_time']) <= v_src,name=f"demand_{k}_src")
                self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[1][0] == k and e[1][2] == self.param['T'] - 1) >= v_dest,name=f"demand_{k}_dest")
        else:
            for i in self.param['sources']:
                self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[0][0] == i and e[0][2] == self.param['min_time']) == self.D[i], name=f"demand_{i}_src")
            for i in self.param['sinks']:
                self.model.addConstr(gp.quicksum(self.x_load[e] for e in self.Ntl.Ntl.edges if e[1][0] == i and e[1][2] == self.param['T'] - 1) == self.D[i], name=f"demand_{i}_dest")

    def construct_model(self, penalty=False):
        self.model = gp.Model()
        self.model.setParam('MIPGap', 0.02)
        self.x_load = dict()
        self.x_ener = dict()
        self.D = dict()
        for i in self.param['sources']:
            self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
        for i in self.param['sinks']:
            self.D[i] = self.model.addVar(lb=0, name=f"D[{i}]")
 
        # create variables
        for e in self.Ntl.Ntl.edges:
            self.x_load[e] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_load', e))
            self.x_ener[e] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=self.__get_var_name('x_ener', e))

        self.n = dict()
        self.a= dict()
        for i in self.N.nodes:
            if i in self.param['tractor_nodes']:
                self.n[i] = self.model.addVar(lb = 0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
            else:
                self.n[i] = self.model.addVar(lb = 0, ub=0, vtype=gp.GRB.INTEGER, name="n[" + str(i) + "]")
            if i in self.param['charge_nodes']:
                self.a[i] = self.model.addVar(lb = 0, vtype=gp.GRB.BINARY, name='a[' + str(i) + ']')
            else:
                self.a[i] = self.model.addVar(lb=0, ub=0,  vtype=gp.GRB.BINARY, name='a[' + str(i) + ']')
        self.model.update()
        
        for v in self.Ntl.Ntl.nodes:
            self.__constr_flow_bal_ener(v)
            self.__constr_flow_bal_load(v)
        for i in self.N.nodes:
            self.__constr_init_cond_x(i)
            self.__constr_end_cond_x(i)
            for t in self.Ntl.times[i]:
                self.__constr_chargecap(i, t)
 
        self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['tractor_nodes']) <= self.param['N_tractors'], name='tractor_limit')
        self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['battery_nodes']) <= self.param['N_batteries'], name='battery_limit')
        self.model.addConstr(gp.quicksum(self.a[i] for i in self.param['charge_nodes'] if i not in self.param.get('existing_charge_nodes', [])) <= self.param['N_chargers'], name='charger_limit')
        self.model.addConstr(gp.quicksum(self.n[i] for i in self.param['charge_nodes']) <= 0, name='charger_limit')

        for e in self.Ntl.Ntl.edges:
            self.__constr_edge(e)
             
        self.__constr_demand()

        if penalty:
            raise NotImplementedError
        self.set_objective()
        self.model.update()

    def set_objective(self):
        obj_expr = gp.quicksum((self.x_load[e]) * self.Ntl.edge_times[e] * self.param['value_for_time'] for e in self.x_load.keys() if e in self.Ntl.Ntl.edges) + gp.quicksum((self.x_ener[e]) * self.Ntl.edge_charges[e] * self.param['charge_cost'][(e[0][0], e[0][2])] for e in self.Ntl.Ntl.edges if self.Ntl.edge_charges[e] > 0) + gp.quicksum(self.param['stat_cost'][i] * self.a[i] + self.param['surplus_cost'][i] * self.n[i] for i in self.N.nodes)
        
        if 'lagrange_multipliers' in self.param.keys():
            obj_expr -= gp.quicksum(self.param['lagrange_multipliers']['load'].get((e[0], e[1]), 0) * self.x_load[e] for e in self.x_load.keys())
            obj_expr -= gp.quicksum(self.param['lagrange_multipliers']['ener'].get((e[0], e[1]), 0) * self.x_ener[e] for e in self.x_ener.keys())

        if 'D' in self.param.keys():
            self.model.setObjective(obj_expr, gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(sum(self.D[i] for i in self.param['sinks']), gp.GRB.MAXIMIZE)

    def update_model(self, penalty=False):
        """Incrementally update the Gurobi model to reflect changes in self.Ntl.Ntl.
        Called instead of construct_model on DDD iterations after the first.
        """
        current_edges = set(self.Ntl.Ntl.edges)
        model_edges   = set(self.x_load.keys())
        new_edges     = current_edges - model_edges
        removed_edges = model_edges   - current_edges

        # 1. Disable variables for removed edges (fix bounds to 0) and remove their edge constraints
        for e in removed_edges:
            self.x_load[e].lb = self.x_load[e].ub = 0
            self.x_ener[e].lb = self.x_ener[e].ub = 0
            for constr_name in [f'e_load_constr[{e}]', f'e_load_ener_bat[{e}]',
                                f'e_load_ener[{e}]', f'e_swap_x[{e}]']:
                c = self.model.getConstrByName(constr_name)
                if c is not None:
                    self.model.remove(c)

        # 2. Add variables for new edges
        for e in new_edges:
            self.x_load[e] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER,
                                               name=self.__get_var_name('x_load', e))
            self.x_ener[e] = self.model.addVar(lb=0, vtype=gp.GRB.INTEGER,
                                               name=self.__get_var_name('x_ener', e))
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
            for name in [f'flow_bal_load[{v}]', f'flow_bal_ener[{v}]']:
                c = self.model.getConstrByName(name)
                if c is not None:
                    self.model.remove(c)
        self.model.update()
        for v in affected_vnodes:
            self.__constr_flow_bal_load(v)
            self.__constr_flow_bal_ener(v)

        # 6. Update init/end condition constraints for affected original nodes
        for i in affected_i:
            for name in [f'init_cond_x[{i}]', f'end_cond_x[{i}]']:
                c = self.model.getConstrByName(name)
                if c is not None:
                    self.model.remove(c)
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
            c = self.model.getConstrByName(f'charge_cap[{i},{t}]')
            if c is not None:
                self.model.remove(c)
        self.model.update()
        for i, t in affected_it:
            if i in self.N.nodes:
                self.__constr_chargecap(i, t)

        # 8. Update demand constraints if source/sink edges changed
        src_changed = any(e[0][0] in self.param['sources'] and e[0][2] == self.param['min_time']  for e in changed_edges)
        snk_changed = any(e[1][0] in self.param['sinks'] and e[1][2] == self.param['T'] - 1  for e in changed_edges)
        if src_changed or snk_changed:
            for name in ['demand-1', 'demand-2']:
                c = self.model.getConstrByName(name)
                if c is not None:
                    self.model.remove(c)
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
        if verbose: print("Infeasible... recomputing full network")
        self.Ntl.construct()
        self.construct_model()
        M = self.model.relax() if LP_relax else self.model
        M.optimize()
                   
        if M.status != gp.GRB.OPTIMAL:
            M.computeIIS()
            M.write('my_iis.ilp')
            print("Irreducible Inconsistent Subsystem (IIS):")
            for c in M.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint: {c.ConstrName}")
            raise RuntimeError("Model is infeasible.")
        if M.status == gp.GRB.OPTIMAL and verbose:
            print("Solution found after full network recomputation!")
            
    
           
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

        try:
            if verbose:
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
                        #if n_iter == 1: raise RuntimeError("Model is infeasible.")
                        self.resolve_infeasibility(LP_relax, verbose)
                        continue
                    if verbose:
                        print("Iteration: ", n_iter)
                        print("Size of network: ", len(self.Ntl.Ntl.nodes), " nodes, ", len(self.Ntl.Ntl.edges), " edges")
                        print("Obj Value: ", M.ObjVal)

                    x_load = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_load" in v.VarName}
                    x_ener = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "x_ener" in v.VarName}
                    a_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "a[" in v.VarName}
                    n_ = {self.parse_var_name(v.VarName): v.X for v in M.getVars() if "n[" in v.VarName}
                    curr_LB = M.ObjVal

                    assert curr_LB >= LB, "objective value too small"
                    print('x_load:', dict((k, v) for k, v in x_load.items() if v > 0))
                    print('x_ener:', dict((k, v) for k, v in x_ener.items() if v > 0))
                    corrected_flow, status = self.Ntl.convert_flow([x_load, x_ener])
                    new_edge, removed_edge = self.Ntl.update([x_load, x_ener])

                    if verbose: print("New edge added: ", new_edge, " Edge removed: ", removed_edge)

                    iter_graph_sizes.append((len(self.Ntl.Ntl.nodes), len(self.Ntl.Ntl.edges)))

                    if penalty:
                        self.penalties[removed_edge] = PEN_CONST
                    else:
                        self.penalties[removed_edge] = 0

                    if status and (new_edge is None):
                        self.Ntl.construct()
                        self.construct_model()
                        model = self.model.relax() if LP_relax else self.model
                        try:
                            model.optimize()
                        except KeyboardInterrupt:
                            _interrupted = True
                            break
                        if model.ObjVal != curr_LB:
                            print("Network was not fully computed, and has been reconstructed")
                            continue
                        if verbose:
                            print("feasible solution found!")
                            print("'''''''''''''''''")
                        avg_time = sum(iter_times) / len(iter_times) if iter_times else 0
                        avg_nodes = sum(s[0] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
                        avg_edges = sum(s[1] for s in iter_graph_sizes) / len(iter_graph_sizes) if iter_graph_sizes else 0
                        print(corrected_flow)
                        return {'x_load': corrected_flow[0], 'x_ener': corrected_flow[1], 'a': a_, 'n':n_}, M.ObjVal, {'n_iterations': n_iter, 'size_of_graph': (len(self.Ntl.Ntl.nodes), len(self.Ntl.Ntl.edges)), 'avg_time_per_iter': avg_time, 'avg_graph_size_per_iter': (avg_nodes, avg_edges)}
                    else:
                        self.update_model(penalty=penalty)


                    if verbose:
                        print("Solution cannot be extended to a feasible solution, updating network...")
                        print({'x_load': corrected_flow[0], 'x_ener': corrected_flow[1]})
                        for i in a_.keys():
                            if a_[i] > 0:
                                print("a[", i, "] = ", a_[i])
                        for i in n_.keys():
                            if n_[i] > 0:
                                print("n[", i, "] = ", n_[i])
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

        def _safe(obj):
            """Recursively make obj JSON-serializable, stringifying unsupported types."""
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_safe(x) for x in obj]
            if isinstance(obj, (bool, type(None))):
                return obj
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (int, float, str)):
                return obj
            return str(obj)

        graph_data = {
            'nodes': [str(n) for n in self.Ntl.Ntl.nodes()],
            'edges': [
                {'u': str(u), 'v': str(v), 'key': k}
                for u, v, k in self.Ntl.Ntl.edges(keys=True)
            ],
            'edge_types':   {str(e): _safe(v) for e, v in self.Ntl.edge_types.items()},
            'edge_times':   {str(e): _safe(v) for e, v in self.Ntl.edge_times.items()},
            'edge_dists':   {str(e): _safe(v) for e, v in self.Ntl.edge_dists.items()},
            'edge_charges': {str(e): _safe(v) for e, v in self.Ntl.edge_charges.items()},
            'charges':      _safe(self.Ntl.charges),
            'times':        _safe(self.Ntl.times),
        }

        snapshot = {
            'param': _safe(self.param),
            'graph': graph_data,
        }

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
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
            print("New variables added: ", list(added_vars)[:l])
        if len(removed_vars) > 0:
            equal = False
            l = min(len(removed_vars), 10)
            print("Variables removed: ", list(removed_vars)[:l])
        if not equal:
            return False 

        old_constrs = set(c.ConstrName for c in old_model.getConstrs())
        new_constrs = set(c.ConstrName for c in self.model.getConstrs())
        added_constrs = new_constrs - old_constrs
        removed_constrs = old_constrs - new_constrs
        if len(added_constrs) > 0:
            equal = False
            print("New constraints added: ", added_constrs)
        if len(removed_constrs) > 0:
            equal = False
            print("Constraints removed: ", removed_constrs)

        return equal
    
    def seed(self, soln):
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
        x_load = soln['x_load']
        x_ener = soln['x_ener']
        modified = False 
        for e in x_load.keys():
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
        for e in x_ener.keys():
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
                print(i)
                print(self.Ntl.charges[i])
                print(self.Ntl.times[i])
            
        self.Ntl.construct()
        self.construct_model()

