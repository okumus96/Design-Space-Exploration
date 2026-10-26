"""
Benders feasibility cuts for the backbone (option configs/buses.json backbone.benders_cuts, 2026-10-26).

The master MILP (optimizer.py) decides placement, cables and candidate paths; it does not hold traffic variables, so its
LP relaxation sees no traffic and the callback teaches it one (flow, path, link) combination per rejected candidate.
This module asks, for every integer candidate, a different question: with the candidate's cables, switches and endpoint
placement, does ANY routing of the point-to-point flows over the candidate paths exist that respects the link
capacities and the chain budgets (transmission and hop times only, queueing relaxed away)? That is a linear program
over path shares f[t, w, p] >= 0:

    sum_{w,p} f[t,w,p]                      >= ind_t            (flow t must be routed; ind_t = 1 for the candidate)
    sum_{t,w,p: link l on path} vol_t f      <= cap_w * n_l      (cables of link l)
    f[t,w,p]                                 <= sw_r             (a relaying path needs a switch at r)
    sum_{t in chain, w, p} lat(t,w,p) f      <= budget - const   (chain network budget)

Solved as a phase-1 LP (minimise the total demand slack). Optimum > 0 = no routing exists. Its optimal duals pi give,
by weak duality, a lower bound on the slack of EVERY master solution y: slack(y) >= sum_t pi_t ind_t(y) + sum_l pi_l
cap_l n_l(y) + sum_r pi_r sw_r(y) + sum_c pi_c B_c, with pi >= 0 on the demand rows and pi <= 0 on the capacity, relay
and chain rows (Gurobi's sign convention for a minimisation). A feasible y needs slack 0, so the right-hand side must
be <= 0: that is the Benders feasibility cut (Costa 2005 for network design). ind_t(y) is the master's linear lower
bound of [flow t active]: src[u] + dst[v] - 1, plus hot[k] - 1 for a standby endpoint that may be cold; pi_t >= 0
keeps the cut valid. One cut covers all links, flows and chains at once; the per-combination cuts stay. Capacity is
per direction (full duplex), as in the callback.

Scope of this version: nominal state only; flows eligible for a bus type are left out (they may be served by a bus
line the LP does not model: a relaxation, so the cut remains valid); queueing terms are not in the LP (relaxation).
"""
import gurobipy as gp
from gurobipy import GRB

from optimizer_utils import robust_ms


def routing_lp_cut(net, ends, cables, switches, comm, hw_use, act_expr):
    """
    One Benders feasibility cut for the candidate, or None if its traffic can be routed.

    ends:     flow id -> (u, v) cells of its endpoints in the candidate (None when an endpoint is unplaced)
    cables:   (a, b, w) -> number of cables in the candidate (a < b)
    switches: set of cells with a SWITCH in the candidate
    comm:     master variables comm[(a, b, w)]; hw_use: master variables hw_use[(j, 'SWITCH')] (may be absent)
    act_expr: flow id -> master expression <= [flow active], equal to 1 at the candidate
    Returns (expr, slack_at_candidate): the cut is expr <= 0, violated by slack_at_candidate > 0 at the candidate.
    """
    bus_types = set(net['bus'].get('types', []))
    flows = []
    for t, (u, v) in ends.items():
        if u is None or v is None or u == v or t not in act_expr:
            continue
        elig = list(net['flow_elig'][t])
        if any(w in bus_types for w in elig):
            continue   # may be served by a bus line: not modelled here, left out (relaxation)
        if net['volumes'][t] <= 0:
            continue
        flows.append((t, u, v, [w for w in elig if w not in bus_types]))
    if not flows:
        return None
    lp = gp.Model('routing_lp')
    lp.Params.OutputFlag = 0
    lp.Params.Threads = 1
    lp.Params.Method = 1
    lp.Params.Presolve = 0
    f, slack = {}, {}
    link_terms, relay_terms, chain_terms = {}, {}, {}
    chain_of_flow = {}
    for ci, ch in enumerate(net['chains']):
        for t in ch['flows']:
            chain_of_flow.setdefault(t, []).append(ci)
    for t, u, v, types in flows:
        pair = (min(u, v), max(u, v))
        for w in types:
            for p, path in enumerate(net['paths'].get(pair + (w,), [])):
                var = lp.addVar(lb=0.0, name=f"f_{t}_{w}_{p}")
                f[t, w, p] = var
                seq = path if (u, v) == pair else path[::-1]
                lat = 0.0
                for a, b in zip(seq, seq[1:]):
                    link_terms.setdefault((a, b, w), []).append((net['volumes'][t], var))   # directed link
                    lat += net['kbit'][t] / net['cap_of'][w] + net['hop_ms'][w]
                for r in path[1:-1]:
                    relay_terms.setdefault(r, []).append(var)
                lat = robust_ms(net, lat)
                for ci in chain_of_flow.get(t, []):
                    chain_terms.setdefault(ci, []).append((lat, var))
    demand_c, cap_c, relay_c, chain_c = {}, {}, {}, {}
    for t, u, v, types in flows:
        vs = [var for (tt, w, p), var in f.items() if tt == t]
        if not vs:
            continue
        slack[t] = lp.addVar(lb=0.0, obj=1.0, name=f"s_{t}")
        demand_c[t] = lp.addConstr(gp.quicksum(vs) + slack[t] >= 1.0, name=f"dem_{t}")
    for (a, b, w), terms in link_terms.items():
        key = (min(a, b), max(a, b), w)
        cap = net['cap_of'][w] * float(cables.get(key, 0))
        cap_c[a, b, w] = lp.addConstr(gp.quicksum(vol * var for vol, var in terms) <= cap, name=f"cap_{a}_{b}_{w}")
    for r, vs in relay_terms.items():
        relay_c[r] = [lp.addConstr(var <= (1.0 if r in switches else 0.0), name=f"relay_{r}_{k}") for k, var in enumerate(vs)]
    for ci, terms in chain_terms.items():
        ch = net['chains'][ci]
        chain_c[ci] = lp.addConstr(gp.quicksum(lat * var for lat, var in terms) <= ch['budget'] - ch['const'], name=f"chain_{ci}")
    lp.ModelSense = GRB.MINIMIZE
    lp.optimize()
    if lp.Status != GRB.OPTIMAL or lp.ObjVal <= 1e-6:
        lp.dispose()
        return None
    # cut: sum pi_row * rhs_row(y) <= 0 (weak duality on the phase-1 LP), violated by ObjVal > 0 at the candidate
    expr, value = gp.LinExpr(), 0.0
    for t, c in demand_c.items():
        mu = max(0.0, c.Pi)
        if mu > 1e-9:
            expr += mu * act_expr[t]
            value += mu
    for (a, b, w), c in cap_c.items():
        lam = min(0.0, c.Pi)
        if lam < -1e-9:
            key = (min(a, b), max(a, b), w)
            expr += lam * net['cap_of'][w] * comm[key]
            value += lam * net['cap_of'][w] * float(cables.get(key, 0))
    for r, cs in relay_c.items():
        nu = sum(min(0.0, c.Pi) for c in cs)
        if nu < -1e-9:
            sw = hw_use.get((r, 'SWITCH'))
            if sw is not None:
                expr += nu * sw
                value += nu * (1.0 if r in switches else 0.0)
            elif r in switches:
                value += nu
    for ci, c in chain_c.items():
        kap = min(0.0, c.Pi)
        if kap < -1e-9:
            ch = net['chains'][ci]
            expr += kap * (ch['budget'] - ch['const'])
            value += kap * (ch['budget'] - ch['const'])
    lp.dispose()
    if value <= 1e-6:   # must be violated at the candidate (= the LP optimum up to rounding); else skip
        return None
    return expr, value


def failure_lp_cut(net, c, ends_all, cables, switches, comm, hw_use, x, r, pos_x, pos_r):
    """
    Benders feasibility cut for the single-failure scenario 'cell c fails' (version 2, 2026-10-26), or None.

    Required traffic after the failure: every base flow between two fail-operational (critical) components must be
    served by a surviving variant (primary-primary, primary-standby, standby-primary or standby-standby, whichever
    endpoints survive) over cables and relays that do not touch c, within the chains' post-failure budgets. The LP
    routes the alive variants of the candidate; a base flow whose alive variant has both ends in one cell needs no
    routing and is left out. The master indicator of a base flow is the candidate's configuration of its four
    instances (x[i,u_i] + x[k,u_k] + r[i,v_i] + r[k,v_k] - 3 <= [same configuration]), so the cut is vacuous for any
    other configuration and therefore valid. Same phase-1 LP and dual-based cut as routing_lp_cut.
    """
    bus_types = set(net['bus'].get('types', []))
    crit = set(net['critical'])
    bases = {}
    for t, tr in net['flow_meta'].items():
        if tr['type'] != 'SC_SC':
            continue
        i, k = tr['src_idx'], tr['dst_idx']
        if i not in crit or k not in crit or net['volumes'][t] <= 0:
            continue
        if any(w in bus_types for w in net['flow_elig'][t]):
            continue
        bases.setdefault(tr.get('base', t), []).append(t)
    required = []   # (base, [(t, u, v, types)] alive variants that need routing, ind terms)
    for b, variants in bases.items():
        tr = net['flow_meta'][variants[0]]
        i, k = tr['src_idx'], tr['dst_idx']
        if pos_x.get(i) is None or pos_x.get(k) is None or pos_r.get(i) is None or pos_r.get(k) is None:
            continue
        alive, trivial = [], False
        for t in variants:
            u, v = ends_all.get(t, (None, None))
            if u is None or v is None or u == c or v == c:
                continue
            if u == v:
                trivial = True
                break
            alive.append((t, u, v, [w for w in net['flow_elig'][t] if w not in bus_types]))
        if trivial or not alive:
            continue
        required.append((b, alive, [x[i, pos_x[i]], x[k, pos_x[k]], r[i, pos_r[i]], r[k, pos_r[k]]]))
    if not required:
        return None
    lp = gp.Model('failure_lp')
    lp.Params.OutputFlag = 0
    lp.Params.Threads = 1
    lp.Params.Method = 1
    lp.Params.Presolve = 0
    f, slack = {}, {}
    link_terms, relay_terms, chain_terms = {}, {}, {}
    chain_of_base = {}
    for ci, ch in enumerate(net['chains']):
        for t in ch['flows']:
            chain_of_base.setdefault(net['flow_meta'][t].get('base', t), []).append(ci)
    base_vars = {}
    for b, alive, _ind in required:
        for t, u, v, types in alive:
            pair = (min(u, v), max(u, v))
            for w in types:
                for p, path in enumerate(net['paths'].get(pair + (w,), [])):
                    if c in path:
                        continue   # a path through the failed cell is dead
                    var = lp.addVar(lb=0.0, name=f"f_{t}_{w}_{p}")
                    f[t, w, p] = var
                    base_vars.setdefault(b, []).append(var)
                    seq = path if (u, v) == pair else path[::-1]
                    lat = 0.0
                    for a, b2 in zip(seq, seq[1:]):
                        link_terms.setdefault((a, b2, w), []).append((net['volumes'][t], var))
                        lat += net['kbit'][t] / net['cap_of'][w] + net['hop_ms'][w]
                    for rr in path[1:-1]:
                        relay_terms.setdefault(rr, []).append(var)
                    lat = robust_ms(net, lat)
                    for ci in chain_of_base.get(b, []):
                        chain_terms.setdefault(ci, []).append((lat, var))
    demand_c, cap_c, relay_c, chain_c = {}, {}, {}, {}
    for b, alive, ind in required:
        slack[b] = lp.addVar(lb=0.0, obj=1.0, name=f"s_{b}")
        demand_c[b] = lp.addConstr(gp.quicksum(base_vars.get(b, [])) + slack[b] >= 1.0, name=f"dem_{b}")
    for (a, b2, w), terms in link_terms.items():
        key = (min(a, b2), max(a, b2), w)
        cap_c[a, b2, w] = lp.addConstr(gp.quicksum(vol * var for vol, var in terms) <= net['cap_of'][w] * float(cables.get(key, 0)), name=f"cap_{a}_{b2}_{w}")
    for rr, vs in relay_terms.items():
        relay_c[rr] = [lp.addConstr(var <= (1.0 if rr in switches else 0.0), name=f"relay_{rr}_{k}") for k, var in enumerate(vs)]
    req_bases = {b for b, _a, _i in required}
    for ci, terms in chain_terms.items():
        ch = net['chains'][ci]
        if all(net['flow_meta'][t].get('base', t) in req_bases or net['flow_meta'][t].get('base', t) in bases for t in ch['flows']):
            chain_c[ci] = lp.addConstr(gp.quicksum(lat * var for lat, var in terms) <= ch['degraded'] - ch['const'], name=f"chain_{ci}")
    lp.ModelSense = GRB.MINIMIZE
    lp.optimize()
    if lp.Status != GRB.OPTIMAL or lp.ObjVal <= 1e-6:
        lp.dispose()
        return None
    expr, value = gp.LinExpr(), 0.0
    ind_of = {b: ind for b, _a, ind in required}
    for b, cst in demand_c.items():
        mu = max(0.0, cst.Pi)
        if mu > 1e-9:
            expr += mu * (gp.quicksum(ind_of[b]) - 3.0)
            value += mu
    for (a, b2, w), cst in cap_c.items():
        lam = min(0.0, cst.Pi)
        if lam < -1e-9:
            key = (min(a, b2), max(a, b2), w)
            expr += lam * net['cap_of'][w] * comm[key]
            value += lam * net['cap_of'][w] * float(cables.get(key, 0))
    for rr, cs in relay_c.items():
        nu = sum(min(0.0, cst.Pi) for cst in cs)
        if nu < -1e-9:
            sw = hw_use.get((rr, 'SWITCH'))
            if sw is not None:
                expr += nu * sw
                value += nu * (1.0 if rr in switches else 0.0)
            elif rr in switches:
                value += nu
    for ci, cst in chain_c.items():
        kap = min(0.0, cst.Pi)
        if kap < -1e-9:
            ch = net['chains'][ci]
            expr += kap * (ch['degraded'] - ch['const'])
            value += kap * (ch['degraded'] - ch['const'])
    lp.dispose()
    if value <= 1e-6:
        return None
    return expr, value
