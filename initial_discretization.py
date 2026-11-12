"""Initial time points for Bi-DDD from the commodities' time windows.

Bi-DDD starts from a uniform grid (min_time, min_time + 1, then every step_size, then
T - 1) at every station. These helpers add, for each commodity k and station i, the
times at which the commodity's load can first be at i and must last leave i, in the
spirit of the significant time points of Shu, Xu and Baldacci (Section 5.2):

  * earliest arrival with charging: a shortest-time search over (station, charge)
    states from (o_k, L) at min_time, with transit (loaded consumption dH), charging at
    charge nodes (ceil(rate * speed) per step, never slower than either network) and
    swaps at swap nodes (to a full battery in the swap time);
  * earliest arrival by travel time alone, min_time + phi(o_k, i): the value the load
    time windows round down to a retained time (Instance._in_time_window), so with it
    retained the window is exact;
  * latest departure, T - 1 - phi(i, z_k).

Points are added only where the commodity's window at i is non-empty. Any set of
points containing min_time and T - 1 gives a valid DDD network, so these only change
how fast Bi-DDD converges, never its result.

Use:
    from initial_discretization import significant_times
    params['initial_discretization'] = {'times': significant_times(N, params)}
    # optionally drop the uniform grid: params['step_size'] = params['T']
"""
import heapq
import math

import networkx as nx


def _segment(curve, g):
    for q in sorted(curve):
        if g < curve[q]['maxbat']:
            return q
    return max(curve)


def earliest_arrival(N, params, origin):
    """{station: earliest time a loaded truck starting full at `origin` at min_time can be
    there}, accounting for battery: transit needs dH, charging and swapping take time."""
    L, T, t0 = params['L'], params['T'], params.get('min_time', 0)
    curve = params['speed_curve']
    rate = params['charge_rate']
    charge_nodes = set(params['charge_nodes'])
    swap_time = {i: params['bat_swap_time'] for i in params['battery_nodes']}
    swap_time.update({i: params['tr_swap_time'] for i in params['tractor_nodes']})

    best = {}
    dist = {(origin, L): t0}
    heap = [(t0, origin, L)]
    while heap:
        t, i, g = heapq.heappop(heap)
        if dist.get((i, g), math.inf) < t or t > T - 1:
            continue
        best[i] = min(best.get(i, math.inf), t)

        def push(j, g2, t2):
            if t2 <= T - 1 and t2 < dist.get((j, g2), math.inf):
                dist[(j, g2)] = t2
                heapq.heappush(heap, (t2, j, g2))

        for j in N.successors(i) if N.is_directed() else N.neighbors(i):
            d = N[i][j]
            if d['dH'] <= g:
                push(j, g - d['dH'], t + d['time'])
        if i in charge_nodes and rate.get(i, 0) > 0:
            q = _segment(curve, g)
            per_step = math.ceil(rate[i] * curve[q]['speed'])   # the fastest either network allows
            for g2 in range(g + 1, min(curve[q]['maxbat'], L) + 1):
                push(i, g2, t + math.ceil((g2 - g) / per_step))
        if i in swap_time and g < L:
            push(i, L, t + swap_time[i])
    return best


def significant_times(N, params):
    """{station: sorted time points} from every commodity's window (see the module docstring)."""
    T, t0 = params['T'], params.get('min_time', 0)
    rev = N.reverse(copy=False) if N.is_directed() else N
    points = {i: set() for i in N.nodes}
    for (o, z) in params['D']:
        travel_from = nx.single_source_dijkstra_path_length(N, o, weight='time')
        travel_to = nx.single_source_dijkstra_path_length(rev, z, weight='time')
        with_charging = earliest_arrival(N, params, o)
        for i in N.nodes:
            if i not in travel_from or i not in travel_to:
                continue                                # k cannot pass through i
            latest = T - 1 - travel_to[i]
            earliest = with_charging.get(i)
            if earliest is None or earliest > latest:
                continue                                # no time at which k can be at i
            points[i].update({t0 + travel_from[i], earliest, latest})
    return {i: sorted(int(t) for t in ts if t0 <= t <= T - 1) for i, ts in points.items() if ts}
