"""
First design by pieces (HANDOFF 16.26): the exact model places the components group by group instead of all at once.
Every step solves the full model of a sub-architecture (the components placed so far plus one new group, with their
messages, devices and chains), the placed components locked to their cells (AssignmentOptimizer.optimize: fixed). A
step is a small solve whatever the size of the car; the result is a complete design that every rule of the model
accepted, checked like any other design (tools/check_solution.py). Settings: configs/solver.json, pieces.

Grouping (deterministic; no learning; no grouping by domain or zone, which would bias the archetype, HANDOFF 16.21):
chains that share a component form one group and a redundant pair stays in one group; groups are ordered by their
tightest network budget, fail-operational groups first; a group above max_group is split chain by chain; a component
outside every chain joins the group of its strongest communication partner while that group has room, the rest form
groups along their messages; the groups are ordered hard first (fail-operational components, heavy message streams);
consecutive groups below min_group are merged. A group without a design (rejected by the
rules given the earlier groups) unlocks the previous group(s), up to max_backtrack, then the build stops. A step is
locked only when it converged (gap below lock_gap): above it the step gets more time (doubled, its own design as the
start, up to piece_time_max_s), and if it still has not converged its group is split in two halves.
"""
import copy
import json
import os
import time
from collections import defaultdict, deque

from optimizer import UNCERTAINTY_BLOCKS, process_memory_gb
from optimizer_utils import is_fail_operational
import optimizer_utils


def _polish_worker(job):
    """One polish solve in its own process (parallel cell polish): installs the cable length table, sends its
    output to its own log file and returns (cell, solution or None, variable values, seconds)"""
    import sys
    t0 = time.time()
    if job.get('distance_table'):
        optimizer_utils.set_distance_table(*job['distance_table'])
    if job.get('steiner_cache'):   # the parent's bus-line tree lengths: no recomputation in this process
        optimizer_utils._STEINER_CACHE.update(job['steiner_cache'])
    if job.get('log'):
        sys.stdout = sys.stderr = open(job['log'], 'w', buffering=1)
    from optimizer import AssignmentOptimizer
    opt = AssignmentOptimizer()
    res = opt.optimize(*job['args'], **job['kwargs'])
    return job['cell'], res, (dict(opt.last_values) if res is not None else None), round(time.time() - t0, 1)

DEFAULTS = {'max_group': 60, 'min_group': 20, 'piece_time_s': 300, 'piece_gap': 0.01, 'piece_phase2_s': 10,
            'max_backtrack': 3, 'lock_gap': 0.10, 'piece_time_max_s': 1200, 'static_capacity': False,
            'polish_rounds': 0, 'polish_time_s': 600, 'polish_partners': 10, 'polish_mode': 'cell', 'polish_max_free': 120, 'polish_parallel': 3, 'polish_passes': 4,
            'polish_stages': None, 'polish_order': 'fewest', 'build_parallel': 1, 'build_lookahead': 8,
            'polish_split_cells': True, 'polish_all': False, 'polish_min_gain': 0.001,
            'step_gurobi_params': 'NoRelHeurTime=20', 'final_network_s': 0, 'final_network_gap': 0.0,
            'gateway_in_build': False}
# gateway_in_build (2026-11-04, default False since the user made the gateway the default rule): False = the group-by-group build runs with the bus gateway off (buses.json
# backbone.bus_gateway) and the gateway enters with the polish and the clean-up. With the gateway the lane endpoints of
# every bus device move with the gateway variables, so the lazy routing and capacity cuts of a step hold for one gateway
# pattern at a time; the r3gwc30 500 build (step 13) and the r3gw1000 build (step 19) found no design in 3 x 1,200 s
# (every candidate rejected, Solution count 0) where their gateway-off twins took 200 s. In the polish all cells but one
# are fixed, so the patterns are few and the gateway model is tractable; the final design is checked under the full model.
# final_network_s 0 = no final network clean-up (_final_network; the runs before 2026-10-30)
# static_capacity off by default (2026-10-29): with the two static cable rules a hard step produced 107 candidates in 900 s,
# all rejected; without them 570 candidates and a design (lab, step 8 of the 300 build)


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def plan_pieces(scs, comm_matrix, chains, fail_cfg, failures_on, cfg):
    """Ordered pieces (lists of component ids) of the build, see the module docstring."""
    ids = [sc.id for sc in scs]
    present = set(ids)
    sc_by_id = {sc.id: sc for sc in scs}
    max_g, min_g = int(cfg['max_group']), int(cfg['min_group'])

    # units = chains (members present in the architecture, as the model reads them) and redundant pairs
    units = []
    for ch in (chains or []):
        members = [s for s in ch.get('swcs', []) if s in present]
        if members:
            units.append((members, float(ch.get('net_budget_ms', float('inf')))))
    for sc in scs:
        p = sc.redundant_with
        if p and p in present and sc.id < p:
            units.append(([sc.id, p], float('inf')))
    uf = _UnionFind(len(units))
    owner = {}
    for k, (members, _b) in enumerate(units):
        for s in members:
            if s in owner:
                uf.union(k, owner[s])
            else:
                owner[s] = k
    groups = defaultdict(list)
    for k in range(len(units)):
        groups[uf.find(k)].append(k)

    def is_crit(members):
        return failures_on and any(is_fail_operational(sc_by_id[s], fail_cfg) for s in members)

    ordered = []
    for ks in groups.values():
        ks = sorted(ks, key=lambda k: (units[k][1], k))   # tightest chain first
        members, seen = [], set()
        for k in ks:
            for s in units[k][0]:
                if s not in seen:
                    seen.add(s)
                    members.append(s)
        ordered.append({'members': members, 'units': ks, 'budget': units[ks[0]][1], 'crit': is_crit(members)})
    ordered.sort(key=lambda g: (not g['crit'], g['budget'], g['members'][0]))

    pieces = []
    for g in ordered:
        if len(g['members']) <= max_g:
            pieces.append(list(g['members']))
            continue
        cur, seen = [], set()   # split chain by chain: a chain joins the piece while its new members fit
        for k in g['units']:
            new = [s for s in units[k][0] if s not in seen]
            if cur and len(cur) + len(new) > max_g:
                pieces.append(cur)
                cur = []
            cur += new
            seen.update(new)
        if cur:
            pieces.append(cur)

    # components outside every chain and pair: the piece of the strongest partner while it has room, else the pool
    in_piece = {s: k for k, p in enumerate(pieces) for s in p}
    vol, nbr = defaultdict(float), defaultdict(set)
    for lk in comm_matrix:
        a, b = lk['src'], lk['dst']
        if a in present and b in present and a != b:
            vol[a, b] += float(lk.get('volume', 0.0))
            vol[b, a] += float(lk.get('volume', 0.0))
            nbr[a].add(b)
            nbr[b].add(a)
    pool = []
    for s in ids:
        if s in in_piece:
            continue
        for _v, t in sorted(((vol[s, t], t) for t in nbr[s] if t in in_piece), reverse=True):
            if len(pieces[in_piece[t]]) < max_g:
                pieces[in_piece[t]].append(s)
                in_piece[s] = in_piece[t]
                break
        else:
            pool.append(s)
    # the pool: groups along the messages, breadth first from the component with the most traffic, capped
    pool_set = set(pool)
    pool.sort(key=lambda s: (-sum(vol[s, t] for t in nbr[s]), s))
    taken = set()
    for seed in pool:
        if seed in taken:
            continue
        cur, q, queued = [], deque([seed]), {seed}
        while q and len(cur) < max_g:
            s = q.popleft()
            cur.append(s)
            taken.add(s)
            for t in sorted(nbr[s] & pool_set, key=lambda t: (-vol[s, t], t)):
                if t not in queued and t not in taken:
                    queued.add(t)
                    q.append(t)
        pieces.append(cur)
    # hard groups first (2026-10-28, after the 300 / 500 runs stalled on late Autoware blocks): the pieces with the
    # most fail-operational components and the heaviest message streams are placed while the car is still empty;
    # flexible components come last. Order is free: a chain is checked once all its members are present.
    total_cpu = sum(sc.cpu_req for sc in scs) or 1
    total_ram = sum(sc.ram_req for sc in scs) or 1

    def hardness(piece):
        # resource-heavy groups first (2026-10-29: the traffic-light containers, 108k DMIPS / 26 GB, placed late forced
        # 2-3 extra cells), then fail-operational count, traffic, tightest chain budget
        heavy = (sum(sc_by_id[s].cpu_req for s in piece) > 0.1 * total_cpu) or (sum(sc_by_id[s].ram_req for s in piece) > 0.1 * total_ram)
        crit = sum(1 for s in piece if is_fail_operational(sc_by_id[s], fail_cfg)) if failures_on else 0
        traffic = sum(vol[s, t] for s in piece for t in nbr[s])
        budget = min((units[k][1] for s in piece if s in owner for k in [owner[s]]), default=float('inf'))
        return ((not heavy) if cfg.get('heavy_first', True) else False, -crit, -traffic, budget)
    pieces.sort(key=hardness)
    # merge consecutive pieces while one of them is small and the pair fits
    merged = []
    for p in pieces:
        if merged and (len(merged[-1]) < min_g or len(p) < min_g) and len(merged[-1]) + len(p) <= max_g:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    assert sorted(s for p in merged for s in p) == sorted(ids), "every component in exactly one piece"
    return merged


def sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep):
    """The architecture restricted to the components keep: their messages, their devices (a device goes with its
    driver), the chains whose members (those present in the architecture) are all in keep; a redundant partner
    outside keep is not yet a constraint."""
    keep = set(keep)
    present = {sc.id for sc in scs}
    sub = []
    for sc in scs:
        if sc.id not in keep:
            continue
        c = copy.copy(sc)
        if c.redundant_with and c.redundant_with not in keep:
            c.redundant_with = None
        sub.append(c)
    comm = [lk for lk in comm_matrix if lk['src'] in keep and lk['dst'] in keep]
    dev_s = {d for sc in sub for d in (sc.sensors or [])}
    dev_a = {d for sc in sub for d in (sc.actuators or [])}
    sens = [s for s in sensors if s.id in dev_s]
    acts = [a for a in actuators if a.id in dev_a]
    chs = [ch for ch in (chains or []) if any(s in present for s in ch.get('swcs', [])) and all(s in keep for s in ch.get('swcs', []) if s in present)]   # a chain without any component of the instance is left out
    return sub, comm, sens, acts, chs


def build_by_pieces(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                    interfaces, backbone, uncertainty, chains, gurobi_params, cfg, phase2_time):
    """One design per robustness level (as optimize_growth_sweep), each built group by group. The run's --time_limit
    is not used: every step gets piece_time_s (times 1 + the number of unlocked groups), the last step the run's
    phase-2 time for the cable length."""
    cfg = {**DEFAULTS, **(cfg or {})}
    if cfg.get('static_capacity', False):
        # every step sees the cable-capacity rules up front (upfront_paths: a flow's path requirement; cell_cut_capacity:
        # per-cell cut-set link capacity), valid rules that are off for the flat model because of its size; a step with
        # a small free part stays small and stops proposing layouts the cables cannot carry (lazy rejections)
        backbone = {**(backbone or {}), 'upfront_paths': True, 'cell_cut_capacity': True}
    params = (uncertainty or {}).get('params', {})
    blocks_on = any((uncertainty or {}).get(k) for k in UNCERTAINTY_BLOCKS)
    levels = [float(a) for a in params.get('rho', {}).get('levels', [1.0])] if blocks_on else [None]
    print("=" * 80)
    print(f"FIRST DESIGN BY PIECES: rho in {levels}, max_group {cfg['max_group']}, min_group {cfg['min_group']}, "
          f"{cfg['piece_time_s']} s per step (--time_limit not used)")
    print("=" * 80)
    solutions = []
    for alpha in levels:
        unc = dict(uncertainty or {})
        if alpha is not None:
            unc['rho'] = alpha
            unc['growth_alpha'] = alpha
            if alpha <= 0.0:   # rho = 0: the deterministic nominal design (optimize_growth_sweep)
                unc['failures'] = False
                unc['redundancy'] = False
            print(f"\n[Pieces] rho = {alpha:.2f}")
        sol = _build_level(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                           hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=unc,
                           chains=chains, gurobi_params=gurobi_params, cfg=cfg, phase2_time=phase2_time)
        if sol is None:
            print(f"[Pieces] rho = {alpha}: no design; sweep stopped")
            break
        solutions.append(sol)
    return solutions


def _build_level(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                 interfaces, backbone, uncertainty, chains, gurobi_params, cfg, phase2_time):
    failures_on = bool(uncertainty.get('failures'))
    fail_cfg = uncertainty.get('params', {}).get('failures', {}) if failures_on else {}
    pieces = plan_pieces(scs, comm_matrix, chains, fail_cfg, failures_on, cfg)
    backbone_full = backbone
    if (backbone or {}).get('bus_gateway') and not cfg.get('gateway_in_build', True):
        backbone = {**backbone, 'bus_gateway': False}   # the build steps: gateway off; polish and clean-up: backbone_full
        print("[Pieces] bus gateway off during the build (pieces.gateway_in_build false); on from the polish", flush=True)
    present = {sc.id for sc in scs}
    by_id = {sc.id: sc for sc in scs}
    traffic = defaultdict(float)
    for lk in comm_matrix:
        traffic[lk['src']] += float(lk.get('volume', 0.0))
        traffic[lk['dst']] += float(lk.get('volume', 0.0))

    def group_info(g):
        return {'fail_op': sum(1 for s in g if failures_on and is_fail_operational(by_id[s], fail_cfg)),
                'traffic_mbit': round(sum(traffic[s] for s in g), 1),
                'devices': sum(len(by_id[s].sensors or []) + len(by_id[s].actuators or []) for s in g),
                'chains': sum(1 for ch in (chains or []) if any(s in g for s in ch.get('swcs', []))),
                'cpu': sum(by_id[s].cpu_req for s in g), 'ram_kb': sum(by_id[s].ram_req for s in g)}
    chain_sets = [{s for s in ch.get('swcs', []) if s in present} for ch in (chains or [])]
    print(f"[Pieces] {len(scs)} components in {len(pieces)} pieces: sizes {[len(p) for p in pieces]}")
    # Gurobi parameters of a step: the run's, then the step's own (later values win, e.g. a short no-relaxation phase)
    step_params = ','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x)
    placed_x, placed_r, steps, sol = {}, {}, [], None
    opt.last_values = None
    k, unlock, t0 = 0, 0, time.time()
    # resume (solver.json pieces.resume_dump, 2026-11-05): continue a stopped build from its last dumped step
    # (dump_dir/pieces_step_<k>.json: the groups and the placements of steps 1..k), the same groups in the same order
    if cfg.get('resume_dump'):
        with open(cfg['resume_dump']) as f:
            dump = json.load(f)
        if {s for p in dump['pieces'] for s in p} != present:
            raise ValueError(f"[Pieces] resume_dump {cfg['resume_dump']}: its groups do not hold the components of this instance")
        pieces = [list(p) for p in dump['pieces']]
        placed_x, placed_r, k = dict(dump['placed_x']), dict(dump['placed_r']), int(dump['step'])
        print(f"[Pieces] resumed from {cfg['resume_dump']}: steps 1..{k} placed ({len(placed_x)} components, "
              f"{len(placed_r)} standbys, cost {float(dump.get('cost') or 0):.2f}), next step {k + 1}/{len(pieces)}", flush=True)
    # parallel build (user 2026-10-29): up to build_parallel groups that exchange no message and share no chain are
    # placed at the same time on the same locked design, each in its own process; a later group (within
    # build_lookahead) is pulled forward only if it is independent of every group it overtakes; one combined solve
    # with all their placements fixed then checks that they fit together; if a group finds no design or the combined
    # solve fails, the batch's groups are placed one by one as before
    msg_nbr = defaultdict(set)
    for lk in comm_matrix:
        if lk['src'] != lk['dst']:
            msg_nbr[lk['src']].add(lk['dst'])
            msg_nbr[lk['dst']].add(lk['src'])
    chains_of = defaultdict(set)
    for ci, cs in enumerate(chain_sets):
        for s_ in cs:
            chains_of[s_].add(ci)

    def independent(a, b):
        sa, sb = set(a), set(b)
        return not any(msg_nbr[s_] & sb for s_ in sa) and not (set().union(*(chains_of[s_] for s_ in sa)) & set().union(*(chains_of[s_] for s_ in sb)))

    dev_obj = {d.id: d for d in list(sensors) + list(actuators)}
    loc_obj = {loc.id: loc for loc in locations}
    pair_vol = defaultdict(float)
    for lk in comm_matrix:
        pair_vol[lk['src'], lk['dst']] += float(lk.get('volume', 0.0))
        pair_vol[lk['dst'], lk['src']] += float(lk.get('volume', 0.0))

    def _own_cells(k0, m):
        """Every group of the batch gets its own cells (user 2026-10-29), so that parallel groups cannot take the same
        cell: a cell is cheap for a group when it is close (cable route) to the devices the group drives and to the
        cells where the group's message partners already sit; cells go to the group they are cheapest for, with a fair
        share each (cells already in use first, as they need no new unit); groups without devices and without placed
        partners get the remaining cells with the least load; every group gets at least two cells (standby)."""
        cells = [loc.id for loc in locations]
        share = max(2, -(-len(cells) // m))
        used = defaultdict(int)
        for s_, c in placed_x.items():
            used[c] += 1
        for s_, c in placed_r.items():
            used[c] += 1
        score = {}
        for i in range(m):
            g = pieces[k0 + i]
            devs = [dev_obj[d] for s_ in g for d in (by_id[s_].sensors or []) + (by_id[s_].actuators or []) if d in dev_obj]
            partners = [(placed_x[t], 1.0 + pair_vol[s_, t]) for s_ in g for t in msg_nbr[s_] if t in placed_x]
            if devs or partners:
                score[i] = {c: sum(optimizer_utils.get_distance(loc_obj[c].location, d.location) for d in devs)
                               + sum(w * optimizer_utils.get_distance(loc_obj[c].location, loc_obj[pc].location) for pc, w in partners)
                            for c in cells}
        own = {i: [] for i in range(m)}
        taken = set()
        # cells in use first, then the others; each cell to the group with the strongest pull that still has room
        for c in sorted(cells, key=lambda c_: (used[c_] == 0, c_)):
            ranked = sorted((score[i][c], i) for i in score if len(own[i]) < share)
            if ranked:
                # a group takes a cell only among its best `share` cells, so that every group keeps its near cells
                for _v, i in ranked:
                    if c in sorted(cells, key=lambda c_: score[i][c_])[:share]:
                        own[i].append(c)
                        taken.add(c)
                        break
        rest = sorted((c for c in cells if c not in taken), key=lambda c_: (used[c_] == 0, used[c_], c_))
        for c in rest:   # groups without a pull, then everyone, round robin by fewest cells
            i = min(range(m), key=lambda i_: (i_ in score, len(own[i_])))
            own[i].append(c)
        for i in range(m):   # at least two cells per group (a standby needs another cell)
            while len(own[i]) < 2:
                donor = max(range(m), key=lambda j: len(own[j]))
                own[i].append(own[donor].pop())
        return own

    def _parallel_batch(k0, m, b_no):
        nonlocal sol, placed_x, placed_r
        import concurrent.futures as cf
        import multiprocessing as mp
        threads = None
        for item in (gurobi_params or '').split(','):
            if item.strip().lower().startswith('threads='):
                threads = int(item.split('=')[1])
        per_job = max(1, (threads or os.cpu_count() or m) // m)
        locked = {s_ for p_ in pieces[:k0] for s_ in p_}
        fixed = {'x': {s_: c for s_, c in placed_x.items() if s_ in locked}, 'r': {s_: c for s_, c in placed_r.items() if s_ in locked}}
        own = _own_cells(k0, m)
        log_dir = cfg.get('dump_dir') or '.'
        os.makedirs(log_dir, exist_ok=True)
        jobs = []
        for i in range(m):
            keep_i = [s_ for p_ in pieces[:k0] for s_ in p_] + list(pieces[k0 + i])
            sub_i, comm_i, sens_i, acts_i, chs_i = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep_i)
            jobs.append({'cell': i, 'log': os.path.join(log_dir, f"build_step{k0 + i + 1}.log"),
                         'distance_table': (list(optimizer_utils._DISTANCE_TABLE[0]), optimizer_utils._DISTANCE_TABLE[1]) if optimizer_utils._DISTANCE_TABLE else None,
                         'steiner_cache': dict(optimizer_utils._STEINER_CACHE),
                         'args': (sub_i, locations, sens_i, acts_i, cable_types, comm_i),
                         'kwargs': dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                                        time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                                        uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                                        phase2_cap=float(cfg['piece_phase2_s']),
                                        gurobi_params=','.join(x for x in (step_params, f'Threads={per_job}') if x),
                                        chains=chs_i, start=opt.last_values,
                                        fixed={**fixed, 'allow': {s_: own[i] for s_ in pieces[k0 + i]}}, lns=None)})
        print(f"\n[Pieces] parallel batch {b_no}: steps {k0 + 1}-{k0 + m} ({[len(pieces[k0 + i]) for i in range(m)]} new components), "
              f"{len(locked)} locked, {m} processes x {per_job} threads, {float(cfg['piece_time_s']):.0f} s; own cells: "
              + "; ".join(f"step {k0 + i + 1}: {', '.join(own[i])}" for i in range(m)), flush=True)
        t_b = time.time()
        with cf.ProcessPoolExecutor(max_workers=m, mp_context=mp.get_context('spawn')) as ex:
            results = list(ex.map(_polish_worker, jobs))
        if any(res_i is None for _i, res_i, _v, _sec in results):
            print(f"[Pieces] parallel batch {b_no}: a group found no design ({[i + k0 + 1 for i, r_, _v, _s in results if r_ is None]}); "
                  f"placing these {m} groups one by one", flush=True)
            return False
        new_x, new_r = dict(fixed['x']), dict(fixed['r'])
        for i, res_i, _v, _sec in results:
            g = set(pieces[k0 + i])
            new_x.update({s_: c for s_, c in res_i['assignment'].items() if s_ in g})
            new_r.update({s_: v['cell'] for s_, v in (res_i.get('standby') or {}).items() if s_ in g})
        keep_m = [s_ for p_ in pieces[:k0 + m] for s_ in p_]
        sub_m, comm_m, sens_m, acts_m, chs_m = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep_m)
        last_m = k0 + m == len(pieces)
        t_c = time.time()
        res_m = opt.optimize(sub_m, locations, sens_m, acts_m, cable_types, comm_m, partitions=partitions, hardwares=hardwares,
                             interfaces=interfaces, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                             uncertainty=uncertainty, backbone=backbone,
                             phase2_time=(phase2_time if last_m else float(cfg['piece_phase2_s'])),
                             phase2_cap=(None if last_m else float(cfg['piece_phase2_s'])),
                             gurobi_params=step_params, chains=chs_m, start=None, fixed={'x': new_x, 'r': new_r}, lns=None)
        if res_m is None:
            print(f"[Pieces] parallel batch {b_no}: the groups do not fit together (combined check found no design in "
                  f"{time.time() - t_c:.0f} s); placing these {m} groups one by one", flush=True)
            return False
        sol = res_m
        placed_x = dict(sol['assignment'])
        placed_r = {s_: v['cell'] for s_, v in (sol.get('standby') or {}).items()}
        ph_m = sol['solve_info']['phases'][0]
        for i, res_i, _v, sec in results:
            ph_i = res_i['solve_info']['phases'][0]
            steps.append({'step': k0 + i + 1, 'new': len(pieces[k0 + i]), 'free': len(pieces[k0 + i]), 'locked': len(locked),
                          'unlocked': 0, 'seconds': sec, **group_info(pieces[k0 + i]), 'result': f'parallel batch {b_no}',
                          'cost': sol.get('total_cost'), 'gap': ph_i.get('mip_gap'), 'cells': len(set(placed_x.values()))})
        if cfg.get('dump_dir'):
            with open(os.path.join(cfg['dump_dir'], f"pieces_step_{k0 + m}.json"), 'w') as f:
                json.dump({'step': k0 + m, 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r,
                           'cost': sol.get('total_cost')}, f, indent=1)
        print(f"[Pieces] parallel batch {b_no} done in {time.time() - t_b:.0f} s (groups {max(r_[3] for r_ in results):.0f} s, "
              f"combined check {time.time() - t_c:.0f} s): cost {sol.get('total_cost'):.2f}, gap of the check "
              f"{100 * (ph_m.get('mip_gap') or 0):.2f} %, {len(set(placed_x.values()))} cells used", flush=True)
        return True

    seq_until, batch_no = 0, 0
    while k < len(pieces):
        if int(cfg.get('build_parallel', 1)) > 1 and unlock == 0 and k >= seq_until:
            window = list(range(k + 1, min(len(pieces), k + int(cfg.get('build_lookahead', 8)))))
            chosen, skipped = [k], []
            for j in window:
                if len(chosen) < int(cfg['build_parallel']) and all(independent(pieces[j], pieces[i]) for i in chosen + skipped):
                    chosen.append(j)
                else:
                    skipped.append(j)
            if len(chosen) > 1:
                pieces[k:] = [pieces[i] for i in chosen] + [pieces[i] for i in range(k, len(pieces)) if i not in chosen]
                m = len(chosen)
                batch_no += 1
                ok = _parallel_batch(k, m, batch_no)
                if ok:
                    k += m
                    continue
                seq_until = k + m
        free_from = max(0, k - unlock)
        keep = [s for p in pieces[:k + 1] for s in p]
        locked = {s for p in pieces[:free_from] for s in p}
        fixed = {'x': {s: c for s, c in placed_x.items() if s in locked},
                 'r': {s: c for s, c in placed_r.items() if s in locked}}
        sub, comm, sens, acts, chs = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep)
        n_free = len(keep) - len(locked)
        last = k == len(pieces) - 1
        t_step = float(cfg['piece_time_s']) * (1 + unlock) * float(cfg.get('_t_scale', 1.0))
        print(f"\n[Pieces] step {k + 1}/{len(pieces)}: {n_free} free components ({len(pieces[k])} new"
              f"{', ' + str(unlock) + ' previous piece(s) unlocked' if unlock else ''}), {len(locked)} locked, "
              f"{len(chs)} chains, {len(sens) + len(acts)} devices, {t_step:.0f} s", flush=True)
        t1 = time.time()
        res = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                           interfaces=interfaces, time_limit=t_step, mip_gap=float(cfg['piece_gap']),
                           uncertainty=uncertainty, backbone=backbone,
                           phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                           phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                           gurobi_params=step_params, chains=chs, start=opt.last_values, fixed=fixed, lns=None, reuse=True)
        step = {'step': k + 1, 'new': len(pieces[k]), 'free': n_free, 'locked': len(locked), 'unlocked': unlock,
                'seconds': round(time.time() - t1, 1), **group_info(pieces[k])}
        if res is None:
            step['result'] = 'no design'
            steps.append(step)
            # no design within the time (2026-11-01, user go): the step's model is kept and solved on, the extra time
            # doubling the total (300 -> 600 -> 1,200 s, up to piece_time_max_s), without rebuilding the model or
            # restarting the search (before, every retry translated the model again and repeated the root work: 2.7 h
            # of the 8.6-h build at 1,000). A step proven infeasible (its rules cannot be met with the locked groups)
            # goes straight to unlocking the previous groups.
            t_used = t_step
            while res is None and t_used < float(cfg['piece_time_max_s']) and getattr(opt, 'last_status', None) != 'INFEASIBLE' \
                    and getattr(opt, '_resumable', False):
                extra = min(t_used, float(cfg['piece_time_max_s']) - t_used)
                print(f"[Pieces] step {k + 1}: no design in {t_used:.0f} s; solving on for {extra:.0f} s more (model kept)", flush=True)
                t1 = time.time()
                res = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                                   interfaces=interfaces, time_limit=extra, mip_gap=float(cfg['piece_gap']),
                                   uncertainty=uncertainty, backbone=backbone,
                                   phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                                   phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                                   gurobi_params=step_params, chains=chs, start=None, fixed=fixed, lns=None, reuse=True, resume=True)
                t_used += extra
                step = {'step': k + 1, 'new': len(pieces[k]), 'free': n_free, 'locked': len(locked), 'unlocked': unlock,
                        'seconds': round(time.time() - t1, 1), 'resumed': True, **group_info(pieces[k])}
                if res is None:
                    step['result'] = 'no design'
                    steps.append(step)
            t_step = t_used
        if res is None:
            unlock += 1
            cfg = {**cfg, '_t_scale': max(1.0, float(cfg['piece_time_max_s']) / float(cfg['piece_time_s']) / (1 + unlock))}
            print(f"[Pieces] step {k + 1}: no design in {t_step:.0f} s; unlocking the previous {unlock} piece(s)", flush=True)
            if unlock > int(cfg['max_backtrack']) or free_from == 0:
                print(f"[Pieces] build stopped at step {k + 1}: no design with {unlock - 1} piece(s) unlocked "
                      f"(max_backtrack {cfg['max_backtrack']})", flush=True)
                return None
            continue
        ph = res['solve_info']['phases'][0]
        # a step is locked only when it converged: above lock_gap it is re-solved with more time (its own design as
        # the start) up to piece_time_max_s; if it still has not converged, its group is split in two halves and
        # the first half is solved again (a group too hard for one step becomes two steps)
        t_budget = t_step
        while (ph.get('mip_gap') or 0.0) > float(cfg['lock_gap']) and t_budget < float(cfg['piece_time_max_s']):
            t_budget = min(2 * t_budget, float(cfg['piece_time_max_s']))
            print(f"[Pieces] step {k + 1}: gap {100 * ph['mip_gap']:.1f} % > {100 * float(cfg['lock_gap']):.0f} %: not locked, "
                  f"re-solving with {t_budget:.0f} s", flush=True)
            again = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                                 interfaces=interfaces, time_limit=t_budget, mip_gap=float(cfg['piece_gap']),
                                 uncertainty=uncertainty, backbone=backbone,
                                 phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                                 phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                                 gurobi_params=step_params, chains=chs, start=opt.last_values, fixed=fixed, lns=None)
            if again is None:
                break
            if again['total_cost'] <= res['total_cost'] + 1e-6:
                res, ph = again, again['solve_info']['phases'][0]
        if (ph.get('mip_gap') or 0.0) > float(cfg['lock_gap']) and len(pieces[k]) >= 2:
            half = len(pieces[k]) // 2
            print(f"[Pieces] step {k + 1}: gap {100 * ph['mip_gap']:.1f} % after {t_budget:.0f} s: group of {len(pieces[k])} "
                  f"split into {half} + {len(pieces[k]) - half}", flush=True)
            pieces[k:k + 1] = [pieces[k][:half], pieces[k][half:]]
            steps.append({**step, 'result': 'split', 'gap': ph.get('mip_gap'), 'seconds': round(time.time() - t1, 1)})
            continue
        sol = res
        cfg = {**cfg, '_t_scale': 1.0}
        placed_x = dict(sol['assignment'])
        placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
        step['seconds'] = round(time.time() - t1, 1)
        if cfg.get('dump_dir'):   # the partial design after every step, to reproduce a failing step on its own
            os.makedirs(cfg['dump_dir'], exist_ok=True)
            with open(os.path.join(cfg['dump_dir'], f"pieces_step_{k + 1}.json"), 'w') as f:
                json.dump({'step': k + 1, 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r,
                           'cost': sol.get('total_cost')}, f, indent=1)
        step.update({'result': ph['status'], 'cost': sol.get('total_cost'), 'gap': ph.get('mip_gap'),
                     'cells': len(set(placed_x.values()))})
        steps.append(step)
        step['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
        print(f"[Pieces] step {k + 1} done in {step['seconds']:.0f} s: cost {step['cost']:.2f}, "
              f"gap {100 * (ph.get('mip_gap') or 0):.2f} %, {step['cells']} cells used; memory {step['memory_gb'][0]:.1f} GB now, "
              f"{step['memory_gb'][1]:.1f} GB peak", flush=True)
        k += 1
        unlock = 0
    build_s = round(time.time() - t0, 1)
    print(f"\n[Pieces] complete design: cost {sol.get('total_cost'):.2f}, {len(pieces)} pieces, "
          f"{sum(1 for s in steps if s['result'] == 'no design')} rejected steps, {sum(1 for s in steps if s['result'] == 'split')} splits, "
          f"{build_s:.0f} s in total")
    polish = []
    backbone = backbone_full
    if int(cfg.get('polish_rounds', 0)) > 0:
        sol, polish = polish_fn(cfg)(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                     hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                     chains=chains, gurobi_params=gurobi_params, cfg=cfg, pieces=pieces, steps=steps, sol=sol,
                                     traffic=traffic)
    final = None
    if float(cfg.get('final_network_s', 0) or 0) > 0:
        sol, final = _final_network(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                    hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                    chains=chains, gurobi_params=gurobi_params, cfg=cfg, sol=sol)
    sol['solve_info']['pieces'] = {'sizes': [len(p) for p in pieces], 'steps': steps, 'build_s': build_s, 'polish': polish,
                                   'final_network': final,
                                   'total_s': round(time.time() - t0, 1), 'settings': {k: v for k, v in cfg.items() if not k.startswith('_')}}
    _report(steps, polish, pieces, sol, cfg, final)
    return sol


def _final_network(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                   backbone, uncertainty, chains, gurobi_params, cfg, sol):
    """Final network clean-up (HANDOFF 16.26, user 2026-10-30): every solve of the build and the polish ends at
    piece_gap of the whole cost or on its time limit, and the network (switches, ports, cables, bus lines) is about
    1 % of that cost, so leftovers of the network survive (500 design: four switch boxes and a switch nobody needed,
    83 USD). One last solve with every primary and standby cell fixed and everything else free (partitions, hardware,
    ports, cables, bus lines), from the current design, to final_network_gap (default 0) within final_network_s; the
    result is kept if it is cheaper, or as cheap with shorter cables. Returns (design, report entry)."""
    t0 = time.time()
    before, before_len = float(sol['total_cost']), float(sol.get('cable_length') or 0.0)
    fixed = {'x': dict(sol['assignment']), 'r': {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}}
    gap = float(cfg.get('final_network_gap', 0.0) or 0.0)
    print(f"\n[Pieces] final network clean-up: every placement fixed, the network free, gap {100 * gap:.2f} %, "
          f"{float(cfg['final_network_s']):.0f} s", flush=True)
    start = dict(opt.last_values or {})
    res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                       interfaces=interfaces, time_limit=float(cfg['final_network_s']), mip_gap=gap,
                       uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                       phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x),
                       chains=chains, start=start, fixed=fixed, lns=None, reuse=True)
    ph = (((res or {}).get('solve_info') or {}).get('phases') or [{}])[0]
    better = res is not None and (float(res['total_cost']) < before - 1e-6 or (float(res['total_cost']) < before + 1e-6 and
                                                                                float(res.get('cable_length') or 0.0) < before_len - 1e-6))
    if better:
        sol = res
    else:
        opt.last_values = start   # the kept design's values, not those of the discarded solve
    switches = sorted(str(h).split('@')[1] for h in sol.get('hw_features', []) if str(h).startswith('SWITCH@'))
    entry = {'before': before, 'after': float(sol['total_cost']), 'seconds': round(time.time() - t0, 1),
             'result': 'improved' if better else ('no gain' if res is not None else 'no design'),
             'status': ph.get('status'), 'gap': ph.get('mip_gap'), 'switches': switches}
    entry['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
    print(f"[Pieces] final network: {before:.2f} -> {entry['after']:.2f} ({entry['result']}; {entry['status']}, gap "
          f"{100 * (entry['gap'] or 0):.3f} %, {entry['seconds']:.0f} s); switches at {switches}", flush=True)
    return sol, entry


def _report(steps, polish, pieces, sol, cfg, final=None):
    """Build report: one line per solve (group properties, time, gap, cost, result), the polish rounds and the final
    network clean-up; printed and, with dump_dir, written as pieces_report.txt"""
    lines = [f"{'step':>4} {'new':>4} {'free':>4} {'lock':>5} {'failop':>6} {'Mbit/s':>8} {'dev':>3} {'chn':>3} {'kDMIPS':>6} {'GB':>5} "
             f"{'s':>6} {'gap %':>6} {'cost':>10} {'cells':>5}  result"]
    for st in steps:
        gap = st.get('gap'); cost = st.get('cost')
        lines.append(f"{st['step']:>4} {st['new']:>4} {st['free']:>4} {st['locked']:>5} {st.get('fail_op', 0):>6} {st.get('traffic_mbit', 0):>8.1f} "
                     f"{st.get('devices', 0):>3} {st.get('chains', 0):>3} {st.get('cpu', 0) / 1000:>6.1f} {st.get('ram_kb', 0) / 1048576:>5.1f} "
                     f"{st['seconds']:>6.0f} {(100 * gap if gap is not None else float('nan')):>6.1f} "
                     f"{(cost if cost is not None else float('nan')):>10.1f} {st.get('cells', '-'):>5}  {st['result']}")
    for r in polish:
        lines.append(f"polish {r['round']} (pass {r.get('pass', 1)}): {r.get('kind') or ('cell' if isinstance(r['group'], str) else 'group')} {r['group']} (+{r['partners']} partners, {r['free']} free): "
                     f"{r['before']:.1f} -> {r['after']:.1f} in {r['seconds']:.0f} s ({r['result']})")
    if final:
        lines.append(f"final network: {final['before']:.1f} -> {final['after']:.1f} in {final['seconds']:.0f} s ({final['result']}; "
                     f"{final['status']}, gap {100 * (final['gap'] or 0):.3f} %), switches at {final['switches']}")
    lines.append(f"final cost {sol.get('total_cost'):.2f}; groups {[len(p) for p in pieces]}")
    text = "\n".join(lines)
    print("\n[Pieces] build report\n" + text, flush=True)
    if cfg.get('dump_dir'):
        os.makedirs(cfg['dump_dir'], exist_ok=True)
        with open(os.path.join(cfg['dump_dir'], 'pieces_report.txt'), 'w') as f:
            f.write(text + "\n")


def _polish_cells_parallel(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                           interfaces, backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    """Cell polish with polish_parallel cells at a time (user 2026-10-29): every batch frees polish_parallel different
    cells of the current design, each in its own process with its own Gurobi session and an equal share of the
    threads, for polish_time_s; the cheapest improvement is kept and the next batch starts from it. A cell that
    improved but was not taken stays a candidate; polish_rounds cells are tried in total (as in the sequential
    polish). Separate processes, not threads: the lazy rule checks run in Python. polish_mode 'level' (user
    2026-10-30, after the bound of the 500 design: 24 partitions where 19 would do by capacity, the small ASIL levels
    spread over many cells, ASIL 3 on 4 cells for 1 % of one partition's load; a partition belongs to one cell and
    one level, and the cell polish cannot move a level's instances off a cell that stays open): the candidates are
    the ASIL levels instead of the cells, every component of the level (primary and standby) freed (_free_sets)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    par = int(cfg.get('polish_parallel', 3))
    max_free = int(cfg.get('polish_max_free', 120))
    threads = None
    for item in (gurobi_params or '').split(','):
        if item.strip().lower().startswith('threads='):
            threads = int(item.split('=')[1])
    per_job = max(1, (threads or os.cpu_count() or par) // par)
    params = ','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params'), f'Threads={per_job}') if x)
    log_dir = cfg.get('dump_dir') or '.'
    os.makedirs(log_dir, exist_ok=True)
    best_values = dict(opt.last_values or {})
    tried, rounds, n_tried, batch = set(), [], 0, 0
    budget = int(cfg['polish_rounds'])
    ctx = mp.get_context('spawn')
    mode, kind = cfg.get('polish_mode', 'cell'), ('level' if cfg.get('polish_mode', 'cell') == 'level' else 'cell')
    while n_tried < budget:
        placed_x = dict(sol['assignment'])
        placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
        cands = _free_sets(mode, sol, scs, cfg, tried, partitions)[:min(par, budget - n_tried)]
        if not cands:
            print(f"[Pieces] parallel {kind} polish: no {kind} left with at most {max_free} components", flush=True)
            break
        batch += 1
        jobs = []
        for n_free, cell, free in cands:
            fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
            jobs.append({'cell': cell, 'n_free': n_free, 'distance_table': (list(optimizer_utils._DISTANCE_TABLE[0]), optimizer_utils._DISTANCE_TABLE[1]) if optimizer_utils._DISTANCE_TABLE else None,
                         'steiner_cache': dict(optimizer_utils._STEINER_CACHE),
                         'log': os.path.join(log_dir, f"polish_batch{batch}_{cell}.log"),
                         'args': (scs, locations, sensors, actuators, cable_types, comm_matrix),
                         'kwargs': dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                                        time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                                        uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                                        phase2_cap=float(cfg['piece_phase2_s']), gurobi_params=params, chains=chains,
                                        start=best_values, fixed=fixed, lns=None)})
        print(f"\n[Pieces] parallel {kind} polish batch {batch}: {[c for _n, c, _f in cands]} ({[n for n, _c, _f in cands]} components freed), "
              f"{len(jobs)} processes x {per_job} threads, {float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
        t1 = time.time()
        with cf.ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:
            results = list(ex.map(_polish_worker, jobs))
        before = float(sol['total_cost'])
        improving = sorted((float(res['total_cost']), cell, res, vals, sec) for cell, res, vals, sec in results
                           if res is not None and float(res['total_cost']) < before - 1e-6)
        taken = improving[0][1] if improving else None
        for cell, res, vals, sec in results:
            n_tried += 1
            entry = {'round': n_tried, 'kind': kind, 'group': cell, 'partners': 0, 'free': next(n for n, c, _f in cands if c == cell),
                     'before': before, 'seconds': sec, 'batch': batch}
            if res is None:
                entry.update({'after': before, 'result': 'no design'})
                tried.add(cell)
            elif float(res['total_cost']) < before - 1e-6:
                entry.update({'after': float(res['total_cost']),
                              'result': 'improved, taken' + _polish_note(mode, cell, sol, res) if cell == taken else 'improved, not taken'})
                if cell == taken:
                    tried.add(cell)
            else:
                entry.update({'after': before, 'result': 'no gain'})
                tried.add(cell)
            rounds.append(entry)
            print(f"[Pieces] polish batch {batch}, {kind} {cell}: {before:.2f} -> {entry['after']:.2f} ({entry['result']}, {sec:.0f} s)", flush=True)
        if improving:
            _c, cell, res, vals, _s = improving[0]
            note = _polish_note(mode, cell, sol, res)
            sol, best_values = res, vals
            if cfg.get('dump_dir'):
                _dump_polish(cfg, pieces, sol, {'stage': mode, 'batch': batch, 'group': cell})
            print(f"[Pieces] batch {batch}: kept {kind} {cell}, {before:.2f} -> {float(sol['total_cost']):.2f}{note}, "
                  f"{time.time() - t1:.0f} s", flush=True)
        else:
            print(f"[Pieces] batch {batch}: no improvement, {time.time() - t1:.0f} s", flush=True)
    opt.last_values = best_values
    return sol, rounds


def _polish_cells(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                  interfaces, backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    """Cell polish (user 2026-10-29): partitions and units are most of the cost, and freeing a group cannot empty a
    cell; so each round frees one whole cell: every component whose primary or standby sits there (both its placement
    and its standby free), the rest of the car fixed; re-solved from the current design for polish_time_s, kept if
    cheaper. Cells in the order of their load (fewest hosted components first: the likeliest to be closed); a cell
    with more than polish_max_free such components is skipped; polish_rounds rounds; a closed cell leaves the list.
    polish_mode 'level': one ASIL level per round instead of one cell (see _polish_cells_parallel). polish_passes
    (2026-10-30, user): the whole list once more after a pass that improved the design, because a move changes what
    the other levels or cells can do (500 run: the ASIL-0 move came in the last round, the four rounds before it saw
    the old positions); at most polish_passes passes, a pass without improvement ends the polish."""
    rounds = []
    best_values = dict(opt.last_values or {})
    max_free = int(cfg.get('polish_max_free', 120))
    mode, kind = cfg.get('polish_mode', 'cell'), ('level' if cfg.get('polish_mode', 'cell') == 'level' else 'cell')
    max_passes, per_pass = max(1, int(cfg.get('polish_passes', 1))), int(cfg['polish_rounds'])
    for n_pass in range(1, max_passes + 1):
        tried, gain = set(), False
        if bool(cfg.get('polish_all', False)):   # 2026-11-02, ablation B: every candidate (cell, level or part) gets a round
            per_pass = max(1, len(_free_sets(mode, sol, scs, cfg, set(), partitions)))
        for r in range(per_pass):
            placed_x = dict(sol['assignment'])
            placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
            cands = _free_sets(mode, sol, scs, cfg, tried, partitions)
            if not cands:
                print(f"[Pieces] {kind} polish: no {kind} left with at most {max_free} components", flush=True)
                break
            n_free, cell, free = cands[0]
            tried.add(cell)
            fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
            t1 = time.time()
            print(f"\n[Pieces] {kind} polish pass {n_pass}, round {r + 1}/{per_pass}: {kind} {cell} ({n_free} components freed), "
                  f"{float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
            # reuse=True (2026-10-30): the whole-car model is translated once, in the first solve of this optimizer on
            # the instance (the rebuild of tools/pieces_polish.py, or this round), and solved again in every later round
            # and in the final network clean-up with only the fixings and the start changed (11-19 min per solve saved)
            res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                               interfaces=interfaces, time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                               uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                               phase2_cap=float(cfg['piece_phase2_s']),
                               gurobi_params=','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x),
                               chains=chains, start=best_values, fixed=fixed, lns=None, reuse=True)
            before = float(sol['total_cost'])
            entry = {'round': len(rounds) + 1, 'pass': n_pass, 'kind': kind, 'group': cell, 'partners': 0, 'free': n_free,
                     'before': before, 'seconds': round(time.time() - t1, 1)}
            if res is not None and float(res['total_cost']) < before - 1e-6:
                note = _polish_note(mode, cell, sol, res)
                sol = res
                best_values = dict(opt.last_values)
                gain = True
                entry.update({'after': float(sol['total_cost']), 'result': 'improved' + note})
                if cfg.get('dump_dir'):
                    _dump_polish(cfg, pieces, sol, {'stage': mode, 'pass': n_pass, 'round': entry['round'], 'group': cell})
            else:
                entry.update({'after': before, 'result': 'no gain' if res is not None else 'no design'})
            entry['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
            rounds.append(entry)
            print(f"[Pieces] {kind} polish round {entry['round']}: {entry['before']:.2f} -> {entry['after']:.2f} ({entry['result']}, "
                  f"{entry['seconds']:.0f} s); memory {entry['memory_gb'][0]:.1f} GB now, {entry['memory_gb'][1]:.1f} GB peak", flush=True)
        if not gain:
            print(f"[Pieces] {kind} polish pass {n_pass}: no improvement, the polish ends", flush=True)
            break
        if n_pass < max_passes:
            print(f"[Pieces] {kind} polish pass {n_pass} improved the design: pass {n_pass + 1} of at most {max_passes}", flush=True)
    opt.last_values = best_values   # the kept design's values (start of the final network clean-up)
    return sol, rounds


def _dump_polish(cfg, pieces, sol, info):
    """pieces_polish.json in dump_dir: the kept design after every improving polish round, in the format of the step
    dumps plus 'polish' (2026-11-01, after the 500 run on the corrected cells died in polish pass 2 and its best design,
    19,227 against 19,908 after the build, was lost): a run that dies in the polish keeps its best design, and
    tools/pieces_polish.py starts from this file when it exists"""
    os.makedirs(cfg['dump_dir'], exist_ok=True)
    placed_x = dict(sol['assignment'])
    placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
    path = os.path.join(cfg['dump_dir'], 'pieces_polish.json')
    with open(path + '.tmp', 'w') as f:
        json.dump({'step': 'polish', 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r, 'cost': sol.get('total_cost'),
                   'cable_length': sol.get('cable_length'), 'polish': {**info, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}},
                  f, indent=1)
    os.replace(path + '.tmp', path)


def _free_sets(mode, sol, scs, cfg, tried, partitions=None):
    """the candidates of a polish round, (size, key, components to free): polish_mode 'cell' = every component with
    its primary or standby on a cell; 'level' = every component of one ASIL level; keys in `tried` and sets above
    polish_max_free left out, except a level, which is cut into parts by cell (each part a candidate, key 'ASIL a part i/n');
    order (cfg polish_order, ablation 2026-10-30): 'fewest' components first (default),
    'most' first, or for levels 'waste' = the level's partitions in the design beyond what its raw load needs
    (largest first; the model's margins are not applied, the order only decides who goes first)"""
    import math
    groups = defaultdict(set)
    if mode == 'level':
        for sc in scs:
            groups[f"ASIL {int(getattr(sc, 'asil_req', 0) or 0)}"].add(sc.id)
    else:
        for s, c in sol['assignment'].items():
            groups[c].add(s)
        for s, v in (sol.get('standby') or {}).items():
            groups[v['cell']].add(s)
    max_free = int(cfg.get('polish_max_free', 120))
    split_cells = bool(cfg.get('polish_split_cells', False))
    if mode == 'level' or split_cells:
        # a level above polish_max_free is cut into parts by cell (2026-11-01, user: ASIL 0 at 1,000 has about 400
        # components and would be skipped); with polish_split_cells (2026-11-02, ablation A: at 1,000 the two biggest
        # cells, above the limit, were skipped) a cell above it is cut into parts by ASIL level the same way; every part
        # gets its own round, the sub-groups spread so the parts are even
        by_id = {sc.id: sc for sc in scs}
        for key in [k for k, v in groups.items() if len(v) > max_free]:
            members = groups.pop(key)
            by_sub = defaultdict(set)
            for sc in members:
                sub = (sol['assignment'].get(sc, '?') if mode == 'level'
                       else f"ASIL {int(getattr(by_id.get(sc), 'asil_req', 0) or 0)}")
                by_sub[sub].add(sc)
            n_parts = max(2, math.ceil(len(members) / max_free))
            parts = [set() for _ in range(n_parts)]
            for sub_members in sorted(by_sub.values(), key=len, reverse=True):
                min(parts, key=len).update(sub_members)
            final_parts = []
            for part in parts:   # a part still above the limit (one sub-group holds more than max_free): cut by id
                n_sub = math.ceil(len(part) / max_free)
                ordered = sorted(part)
                final_parts += [set(ordered[j::n_sub]) for j in range(n_sub)] if n_sub > 1 else [part]
            for i, part in enumerate(final_parts, 1):
                groups[f"{key} part {i}/{len(final_parts)}"] = part
            print(f"[Pieces] {mode} polish: {key} ({len(members)} components) above polish_max_free {max_free}: "
                  f"{len(final_parts)} parts of {[len(q) for q in final_parts]} by {'cell' if mode == 'level' else 'level'}", flush=True)
    items = [(len(v), k, v) for k, v in groups.items() if k not in tried and len(v) <= max_free]
    order = str(cfg.get('polish_order', 'fewest') or 'fewest')
    if mode == 'level' and order == 'partitions':   # 2026-11-02, ablation B: the level (or part) with the most
        # partitions in the design first, then fewest components; a merge is likeliest where partitions are many
        used = defaultdict(int)
        for key, n in (sol.get('partition_counts') or {}).items():
            used['ASIL ' + key.rsplit('_asil', 1)[1]] += int(n)
        return sorted(items, key=lambda t: (-used.get(' '.join(t[1].split()[:2]), 0), t[0], t[1]))
    if mode == 'level' and order == 'waste':
        by_id = {sc.id: sc for sc in scs}
        caps = {r: float((partitions or {}).get(f'{r}_cap', 0) or 0) for r in ('cpu', 'ram', 'rom')}
        used = defaultdict(int)
        for key, n in (sol.get('partition_counts') or {}).items():
            used['ASIL ' + key.rsplit('_asil', 1)[1]] += int(n)

        def need(members):
            n = 1
            for r, cap in caps.items():
                if cap > 0:
                    n = max(n, math.ceil(sum(float(getattr(by_id[s], f'{r}_req', 0) or 0) for s in members) / cap - 1e-9))
            return n
        return sorted(items, key=lambda t: (-(used.get(t[1], 0) - need(t[2])), t[0], t[1]))
    return sorted(items, key=lambda t: ((-t[0] if order == 'most' else t[0]), t[1]))


def _polish_stages(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                   backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    """Polish in stages (ablation 2026-10-30): cfg polish_stages = a list of polish modes, e.g. ['cell', 'level'], run
    one after another, each as one pass of the sequential polish (model reused); the whole list again while a full
    round of stages improved the design, at most polish_passes times."""
    rounds = []
    stages = list(cfg.get('polish_stages') or [cfg.get('polish_mode', 'cell')])
    max_passes = max(1, int(cfg.get('polish_passes', 1)))
    for n_pass in range(1, max_passes + 1):
        before = float(sol['total_cost'])
        for stage in stages:
            print(f"\n[Pieces] polish pass {n_pass} of at most {max_passes}, stage {stage}", flush=True)
            sol, rs = _polish_cells(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                    hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                    chains=chains, gurobi_params=gurobi_params, pieces=pieces, steps=steps, sol=sol, traffic=traffic,
                                    cfg={**cfg, 'polish_mode': stage, 'polish_passes': 1, 'polish_parallel': 1})
            for r in rs:
                r.update({'round': len(rounds) + 1, 'pass': n_pass, 'stage': stage})
                rounds.append(r)
        after, min_gain = float(sol['total_cost']), float(cfg.get('polish_min_gain', 0.0) or 0.0)
        if after >= before - 1e-6:
            print(f"[Pieces] polish pass {n_pass}: no stage improved the design, the polish ends", flush=True)
            break
        if min_gain > 0 and before - after < min_gain * before:   # 2026-11-02, ablation C: a pass gaining less than
            # polish_min_gain of the cost ends the polish (with more passes allowed than before)
            print(f"[Pieces] polish pass {n_pass}: gain {before:.2f} -> {after:.2f} below polish_min_gain {100 * min_gain:.2f} %, "
                  f"the polish ends", flush=True)
            break
        if n_pass < max_passes:
            print(f"[Pieces] polish pass {n_pass} improved the design {before:.2f} -> {float(sol['total_cost']):.2f}: "
                  f"pass {n_pass + 1} of at most {max_passes}", flush=True)
    return sol, rounds


def _polish_note(mode, key, old, new):
    """what a kept polish solve changed: 'cell' = whether the cell closed; 'level' = the level's partitions and the
    total before and after"""
    if mode == 'level':
        a = key.split()[1]   # 'ASIL 0' or 'ASIL 0 part 1/2'
        return (f", partitions {_level_partitions(old, a)} -> {_level_partitions(new, a)} of the level, "
                f"{int(old.get('num_partitions', 0))} -> {int(new.get('num_partitions', 0))} in total")
    used = set(new['assignment'].values()) | {v['cell'] for v in (new.get('standby') or {}).values()}
    return ', cell closed' if key.split(' part ')[0] not in used else ''


def _level_partitions(sol, a):
    """partitions of ASIL level a in a design, over all cells"""
    return sum(int(n) for k, n in (sol.get('partition_counts') or {}).items() if k.endswith(f'_asil{a}'))


def polish_fn(cfg):
    """the polish of polish_mode: 'cell' (one cell per round) and 'level' (one ASIL level per round) share the cell
    polish code, polish_parallel candidates at a time; polish_stages = several modes one after another
    (_polish_stages); anything else = the group polish"""
    if cfg.get('polish_stages'):
        return _polish_stages
    if cfg.get('polish_mode', 'cell') in ('cell', 'level'):
        return _polish_cells_parallel if int(cfg.get('polish_parallel', 1)) > 1 else _polish_cells
    return _polish


def _polish(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
            backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    """Polish rounds (user 2026-10-29): the whole car in the model, everything fixed except one group and its
    strongest communication partners; re-solved from the current design for polish_time_s; kept if cheaper. Groups
    in the order of their build gap (weakest first), then by cost; polish_rounds rounds in total."""
    gap_of = {}
    for st in steps:
        if st['result'] not in ('no design', 'split') and st.get('gap') is not None:
            gap_of[st['step'] - 1] = st['gap']
    order = sorted(range(len(pieces)), key=lambda k: -gap_of.get(k, 0.0))
    n_partners = int(cfg.get('polish_partners', 10))
    pair = defaultdict(float)
    for lk in comm_matrix:
        pair[lk['src'], lk['dst']] += float(lk.get('volume', 0.0))
        pair[lk['dst'], lk['src']] += float(lk.get('volume', 0.0))
    nbr = defaultdict(set)
    for a, b in pair:
        nbr[a].add(b)
    best_values = dict(opt.last_values or {})
    placed_x = dict(sol['assignment'])
    placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
    rounds = []
    for r in range(int(cfg['polish_rounds'])):
        g = order[r % len(order)]
        group = set(pieces[g])
        partners = sorted(((sum(pair[s, t] for s in group if (s, t) in pair), t) for t in {t for s in group for t in nbr[s]} - group), reverse=True)
        free = group | {t for _v, t in partners[:n_partners]}
        fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
        t1 = time.time()
        print(f"\n[Pieces] polish round {r + 1}/{cfg['polish_rounds']}: group {g + 1} (build gap {100 * gap_of.get(g, 0):.1f} %), "
              f"{len(free)} free ({len(free) - len(group)} partners), {float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
        res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                           interfaces=interfaces, time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                           uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                           gurobi_params=gurobi_params, chains=chains, start=best_values, fixed=fixed, lns=None)
        before = float(sol['total_cost'])
        entry = {'round': r + 1, 'group': g + 1, 'partners': len(free) - len(group), 'free': len(free), 'before': before,
                 'seconds': round(time.time() - t1, 1)}
        if res is not None and float(res['total_cost']) < before - 1e-6:
            sol = res
            best_values = dict(opt.last_values)
            placed_x = dict(sol['assignment'])
            placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
            entry.update({'after': float(sol['total_cost']), 'result': 'improved'})
        else:
            entry.update({'after': before, 'result': 'no gain' if res is not None else 'no design'})
        rounds.append(entry)
        print(f"[Pieces] polish round {r + 1}: {entry['before']:.2f} -> {entry['after']:.2f} ({entry['result']}, {entry['seconds']:.0f} s)", flush=True)
    return sol, rounds
