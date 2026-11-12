"""Initial time points for Bi-DDD from the commodities' time windows.

Bi-DDD starts from a uniform grid (min_time, min_time + 1, then every step_size, then
T - 1) at every station. These helpers add, for each commodity k and station i, the
times at which the commodity's load can first be at i and must last leave i, in the
spirit of the significant time points of Shu, Xu and Baldacci (Section 5.2).

The times come from shortest-time searches over (station, battery level) states for a
loaded truck (consumption dH) that may recharge by charging at charge nodes
(ceil(rate * speed) levels per step within a segment, never slower than either network
allows) and by swapping at swap nodes (to a full battery in the swap time).

Points, for each commodity k and station i:
  * earliest arrival: the truck leaves o_k full at min_time; the earliest time it can
    be at i;
  * latest departure: the latest time it can leave i (full, which is never worse) and
    still reach z_k by T - 1;
  * at charge nodes, the earliest time the truck can be at i with a full battery, when
    that is later than its earliest arrival (the heuristic network's "charge to full").
Points are added only where the window is non-empty (earliest arrival no later than
latest departure), and, with max_detour, only at stations on a schedule at most that many
steps slower than the commodity's fastest (through_times). Any set of points containing min_time and T - 1 gives a valid DDD
network, so these only change how fast Bi-DDD converges, never its result.

Earlier versions also added the travel-time window ends, separate points for charging
only and swapping only, and a departure for each leg out of a charge node; that gave
about twice the edges of the uniform grid on the full case study.

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


class _Rules:
    """Recharging rules for one mode."""

    def __init__(self, params, charge, swap):
        self.L, self.T = params['L'], params['T']
        self.curve = params['speed_curve']
        self.charge_rate = {i: r for i, r in params['charge_rate'].items()
                            if charge and i in set(params['charge_nodes']) and r > 0}
        swap_time = {i: params['bat_swap_time'] for i in params['battery_nodes']}
        swap_time.update({i: params['tr_swap_time'] for i in params['tractor_nodes']})
        self.swap_time = swap_time if swap else {}

    def per_step(self, i, g):
        """Levels gained per step charging at i from level g, and the segment's top level."""
        q = _segment(self.curve, g)
        return math.ceil(self.charge_rate[i] * self.curve[q]['speed']), min(self.curve[q]['maxbat'], self.L)


def _out(N, i):
    return N.successors(i) if N.is_directed() else N.neighbors(i)


def _in(N, j):
    return N.predecessors(j) if N.is_directed() else N.neighbors(j)


def _forward(N, params, origin, charge, swap):
    """{(station, level): earliest time} for a loaded truck leaving `origin` full at min_time."""
    rules, t0 = _Rules(params, charge, swap), params.get('min_time', 0)
    L, T = rules.L, rules.T
    dist = {(origin, L): t0}
    heap = [(t0, origin, L)]

    def push(j, g2, t2):
        if t2 <= T - 1 and t2 < dist.get((j, g2), math.inf):
            dist[(j, g2)] = t2
            heapq.heappush(heap, (t2, j, g2))

    while heap:
        t, i, g = heapq.heappop(heap)
        if dist.get((i, g), math.inf) < t:
            continue
        for j in _out(N, i):
            if N[i][j]['dH'] <= g:
                push(j, g - N[i][j]['dH'], t + N[i][j]['time'])
        if i in rules.charge_rate:
            per, top = rules.per_step(i, g)
            for g2 in range(g + 1, top + 1):
                push(i, g2, t + math.ceil((g2 - g) / per))
        if i in rules.swap_time and g < L:
            push(i, L, t + rules.swap_time[i])
    return dist


def _backward(N, params, dest, charge, swap):
    """{(station, level): fastest time from that state to `dest`} (the same moves as
    _forward, searched backwards from every state at `dest`)."""
    rules = _Rules(params, charge, swap)
    L = rules.L
    togo = {(dest, g): 0 for g in range(L + 1)}
    heap = [(0, dest, g) for g in range(L + 1)]

    def push(i, g, d):
        if d < togo.get((i, g), math.inf):
            togo[(i, g)] = d
            heapq.heappush(heap, (d, i, g))

    while heap:
        d, j, h = heapq.heappop(heap)
        if togo.get((j, h), math.inf) < d:
            continue
        for i in _in(N, j):                              # transit i -> j arriving with h
            g = h + N[i][j]['dH']
            if g <= L:
                push(i, g, d + N[i][j]['time'])
        if j in rules.charge_rate:                       # charging at j from g up to h
            for g in range(h):
                per, top = rules.per_step(j, g)
                if h <= top:
                    push(j, g, d + math.ceil((h - g) / per))
        if j in rules.swap_time and h == L:              # swapping at j to a full battery
            for g in range(L):
                push(j, g, d + rules.swap_time[j])
    return togo


def earliest_arrival(N, params, origin, charge=True, swap=True):
    """{station: earliest time a loaded truck starting full at `origin` at min_time can be
    there}, recharging by charging (if `charge`) and by swapping (if `swap`)."""
    best = {}
    for (i, _), t in _forward(N, params, origin, charge, swap).items():
        best[i] = min(best.get(i, math.inf), t)
    return best


def latest_departure(N, params, dest, charge=True, swap=True):
    """{station: latest time a loaded truck can leave it, full, and reach `dest` by T - 1}."""
    T, L = params['T'], params['L']
    return {i: T - 1 - d for (i, g), d in _backward(N, params, dest, charge, swap).items()
            if g == L and T - 1 - d >= params.get('min_time', 0)}


def departures_after_charging(N, params, i, forward, targets=None):
    """Earliest times at which the truck can be at charge node i holding at least each
    target level (default: L, a full battery), from a `forward` search."""
    at_i = {g: t for (j, g), t in forward.items() if j == i}
    if not at_i:
        return set()
    times = set()
    for target in (targets or {params['L']}):
        reach = [t for g, t in at_i.items() if g >= target]
        if reach:
            times.add(min(reach))
    return times


def through_times(N, params, origin, dest, forward=None):
    """{station: duration of the fastest schedule from `origin` (full, at min_time) to `dest`
    that passes through it}, respecting the battery: the minimum over battery levels g of
    the earliest time at (station, g) plus the fastest time from (station, g) to `dest`,
    minus min_time. Charging and swapping are both allowed."""
    t0 = params.get('min_time', 0)
    fwd = forward if forward is not None else _forward(N, params, origin, charge=True, swap=True)
    togo = _backward(N, params, dest, charge=True, swap=True)
    best = {}
    for (i, g), t in fwd.items():
        if (i, g) in togo:
            best[i] = min(best.get(i, math.inf), t - t0 + togo[(i, g)])
    return best


def significant_times(N, params, max_detour=None):
    """{station: sorted time points} from every commodity's window (see the module docstring).

    max_detour: if set (in time steps), a commodity's points are added only at stations
    on a schedule at most max_detour steps slower than its fastest one, i.e. where
    through_times <= fastest + max_detour; other stations keep only the base grid.
    """
    T, t0 = params['T'], params.get('min_time', 0)
    charge_nodes = set(params['charge_nodes'])
    points = {i: set() for i in N.nodes}
    for (o, z) in params['D']:
        fwd = _forward(N, params, o, charge=True, swap=True)
        arrive = {}
        for (i, _), t in fwd.items():
            arrive[i] = min(arrive.get(i, math.inf), t)
        leave = latest_departure(N, params, z, charge=True, swap=True)
        near = None
        if max_detour is not None:
            through = through_times(N, params, o, z, forward=fwd)
            if not through:
                continue                                    # z unreachable: no points
            fastest = min(through.values())
            near = {i for i, d in through.items() if d <= fastest + max_detour}
        for i in N.nodes:
            if near is not None and i not in near:
                continue                                    # far from k's fastest schedules
            if i not in arrive or i not in leave or arrive[i] > leave[i]:
                continue                                    # no time at which k can be at i
            points[i].update({arrive[i], leave[i]})
            if i in charge_nodes:
                points[i].update(t for t in departures_after_charging(N, params, i, fwd)
                                 if arrive[i] < t <= leave[i])
    return {i: sorted(int(t) for t in ts if t0 <= t <= T - 1) for i, ts in points.items() if ts}
