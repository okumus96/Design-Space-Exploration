"""
Learned guide of the exact search (HANDOFF 16.25), built by main.py when configs/solver.json learn.search is on.

mode 'rl' (the method): the placement head proposes a start design (a MIP start only, nothing is excluded), phase 1 is
the exact model for init_s seconds; then the destroy policy (RLPolicy) picks in every iteration the components to free
from the current design, the exact model re-solves them with the rest fixed, and the result is the next design.
mode 'predict' (the earlier supervised variant): phase 1 only within the top_k predicted cells per component, then
neighbourhoods around uncertain predictions. Controls of 'rl' with the same start and loop: mode 'random' frees
components drawn at random, 'cells' whole cells of the current design plus communication partners, 'cluster' a
breadth-first cluster of the communication graph. In every mode every design passes the lazy constraints of the full
model.

Start (fix_start, default on; method 3 part 1 of HANDOFF 16.23): the predicted placement is repaired for the capacity
rules (repair_placement) and fixed in the first solve, so the exact model only fills in partitions, hardware, network
and cables (seconds instead of the full search for a first design); the placement is freed again for the loop. With
fix_start off the prediction is only a MIP start hint of the full search (the earlier behaviour).
"""
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from learn.features import InstanceGraph, load_instance                     # noqa: E402
from learn.model import PolicyNet, plackett_luce_logp, predict, sample_free   # noqa: E402


def load_policy(path):
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    ck = torch.load(path, weights_only=False, map_location='cpu')
    model = PolicyNet.from_cfg(ck['cfg'])
    model.load_state_dict(ck['state'])
    model.eval()
    return model, ck


class RLPolicy:
    """Destroy policy for optimizer.py's improvement loop: from the current design it picks free_share of the
    components (sampled without replacement from the destroy logits, or the top ones when greedy) and records state,
    action, log probability, value and the reward of every iteration (cost reduction in % of the start cost)."""

    def __init__(self, model, ig, max_iters, seed=0, greedy=False, temperature=1.0):
        self.model, self.ig = model, ig
        self.max_iters, self.greedy, self.temp = max(1, int(max_iters or 1)), greedy, float(temperature)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.steps, self.last_freed, self.start_cost = [], [], None

    def __call__(self, rng, free_share, design=None):
        it = len(self.steps)
        it_frac = min(1.0, it / self.max_iters)
        with torch.no_grad():
            out = self.model(self.ig.data(design, self.last_freed, it_frac))
        logits = out['free'] / self.temp
        k = max(1, int(round(free_share * logits.numel())))
        order = torch.topk(logits, k).indices if self.greedy else sample_free(logits, k, self.gen)
        freed = [self.ig.comp_ids[int(i)] for i in order]
        self.steps.append({'design': design, 'freed_last': list(self.last_freed), 'it_frac': it_frac,
                           'order': order.tolist(), 'logp': float(plackett_luce_logp(logits, order)),
                           'value': float(out['value'][0])})
        self.last_freed = freed
        return set(freed)

    def observe(self, before, after):
        if self.start_cost is None:
            self.start_cost = before
        self.steps[-1]['reward'] = 100.0 * (before - after) / max(1e-9, abs(self.start_cost))


def repair_placement(ig, px, pr):
    """The predicted placement made admissible for the capacity rules of the exact model (the prediction alone can
    overload a cell many times). Primary cell = the most probable cell with room; room = every resource of the cell's
    components plus its hot standbys within the cell's capacity, every (cell, ASIL) group within max_per_asil
    partitions, the cell within site_max_partitions, and the same after the failure of any other cell (the cold
    standbys that come alive). Components leave an overloaded cell least confident first. Standby = the most
    probable other cell that keeps the failure rule. Returns (start, start_standby, confidence, moves). The rules
    of the exact model (chains, buses, routing, packing) are left to the solve with this placement fixed."""
    import math
    n, m = len(ig.comp_ids), len(ig.cell_ids)
    dem, cap = ig.demand.tolist(), ig.cap_units.tolist()
    cell_of = {c: j for j, c in enumerate(ig.cell_ids)}
    order = {i: sorted(px[c], key=px[c].get, reverse=True) for i, c in enumerate(ig.comp_ids)}
    x = {i: cell_of[order[i][0]] for i in range(n)}
    r = {}

    def partitions(load_by_asil):   # partitions a cell needs for the group loads {asil: [cpu, ram, rom]}
        need = {a: max(math.ceil(v - 1e-9) for v in load[:3]) for a, load in load_by_asil.items()}
        return need

    def loads(j, extra=(), without=None):
        """resource loads of cell j: its components, its hot standbys and the components of `extra` (activated cold
        standbys); returns (total [cpu, ram, rom, tops], {asil: [cpu, ram, rom]})"""
        tot, by_a = [0.0] * 4, {}
        members = [i for i in range(n) if x[i] == j and i != without] + \
                  [i for i in range(n) if ig.hot_forced[i] and r.get(i) == j and i != without] + list(extra)
        for i in members:
            for q in range(4):
                tot[q] += dem[i][q]
            g = by_a.setdefault(ig.asil[i], [0.0, 0.0, 0.0])
            for q in range(3):
                g[q] += dem[i][q]
        return tot, by_a

    def fits(j, extra=(), without=None):
        tot, by_a = loads(j, extra, without)
        if any(tot[q] > cap[j][q] + 1e-9 for q in range(4)):
            return False
        need = partitions(by_a)
        if any(v > ig.max_per_asil for v in need.values()):
            return False
        if ig.site_max_partitions is not None and sum(need.values()) > ig.site_max_partitions:
            return False
        return True

    def fits_after_failures(j, extra=(), without=None):
        """fits, nominally and after the failure of any other cell (its cold standbys at j come alive)"""
        if not fits(j, extra, without):
            return False
        for c in range(m):
            if c == j:
                continue
            woken = [i for i in range(n) if x[i] == c and r.get(i) == j and not ig.hot_forced[i] and i != without]
            if woken and not fits(j, list(extra) + woken, without):
                return False
        return True

    partner = {}
    for i, k in ig.redundant:
        partner.setdefault(i, set()).add(k)
        partner.setdefault(k, set()).add(i)

    def apart(i, j):   # redundant partners of i are not at j
        return all(x[k] != j for k in partner.get(i, ()))

    def move(i, avoid):
        """i to its most probable cell other than `avoid` with room and apart from its partners; else the emptiest"""
        for cid in order[i]:
            j = cell_of[cid]
            if j != avoid and apart(i, j) and fits(j, extra=[i], without=i):
                x[i] = j
                return
        x[i] = min((j for j in range(m) if j != avoid), key=lambda j: sum(loads(j, without=i)[0]))

    moves = 0
    # chains first: a chain's network latency budget allows few cell crossings (the model's chain bounds), so every
    # chain gathers in the cell that holds most of its predicted probability mass, as far as the cell has room;
    # components of several chains follow the longest chain
    in_chain = set()
    for members in sorted(ig.chains, key=len, reverse=True):
        members = [i for i in members if i not in in_chain]
        if not members:
            continue
        mass = {}
        for i in members:
            for cid, p in px[ig.comp_ids[i]].items():
                mass[cell_of[cid]] = mass.get(cell_of[cid], 0.0) + p
        for j in sorted(mass, key=mass.get, reverse=True):
            moved_here = [i for i in members if x[i] == j]
            for i in sorted((i for i in members if x[i] != j), key=lambda i: -px[ig.comp_ids[i]][ig.cell_ids[j]]):
                if apart(i, j) and fits(j, extra=[i], without=i):
                    x[i] = j
                    moved_here.append(i)
                    moves += 1
            if len(moved_here) >= len(members) or sum(mass.values()) == 0:
                break
            members = [i for i in members if i not in moved_here]   # the rest gathers in the next cell
        in_chain.update(moved_here)
        in_chain.update(members)
    for _ in range(10 * n):   # primaries: empty every overloaded cell (least confident component first), part every pair
        bad = next((j for j in range(m) if not fits(j)), None)
        pair = next(((i, k) for i, k in ig.redundant if x[i] == x[k]), None)
        if bad is not None:   # chain members stay as long as others can go
            i = min((i for i in range(n) if x[i] == bad), key=lambda i: (i in in_chain, px[ig.comp_ids[i]][ig.cell_ids[bad]]))
        elif pair is not None:
            i = min(pair, key=lambda i: (i in in_chain, px[ig.comp_ids[i]][ig.cell_ids[x[i]]]))
        else:
            break
        move(i, x[i])
        moves += 1
    # balance for failures: when a cell fails, the standbys of its critical components come alive, and a chain with a
    # tight budget then allows at most one crossing, so they must fit together into one other cell (capacity and
    # partition limits as in the model; what the model's capacity-after-failure and degraded chain bounds require of
    # the best designs). The least confident critical component of a cell that is too heavy moves to the cell with
    # the most room.
    def headroom(j, extra=()):
        tot = loads(j, extra)[0]
        return [cap[j][q] - tot[q] for q in range(4)]

    for _ in range(10 * n):
        heavy = None
        for c in range(m):
            crit = [i for i in range(n) if x[i] == c and ig.critical[i]]
            if crit and not any(fits(j, extra=crit) for j in range(m) if j != c):
                heavy = c
                break
        if heavy is None:
            break
        i = min((i for i in range(n) if x[i] == heavy and ig.critical[i]), key=lambda i: px[ig.comp_ids[i]][ig.cell_ids[heavy]])
        cands = [j for j in range(m) if j != heavy and apart(i, j) and fits(j, extra=[i], without=i)]
        if not cands:
            break
        x[i] = max(cands, key=lambda j: min(headroom(j, extra=[i])))
        moves += 1
    for i in sorted(range(n), key=lambda i: -sum(dem[i])):   # standbys of the critical components, largest first
        if not ig.critical[i]:
            continue
        c = ig.comp_ids[i]
        for cid in sorted(pr[c], key=pr[c].get, reverse=True):
            j = cell_of[cid]
            if j != x[i]:
                r[i] = j
                if fits_after_failures(j):
                    break
                del r[i]
        if i not in r:   # no cell keeps the failure rule: the most probable other cell, the solve may still repair it
            r[i] = cell_of[next(cid for cid in sorted(pr[c], key=pr[c].get, reverse=True) if cell_of[cid] != x[i])]
    start = {ig.comp_ids[i]: ig.cell_ids[x[i]] for i in range(n)}
    start_r = {ig.comp_ids[i]: ig.cell_ids[r[i]] for i in r}
    conf = {c: px[c][start[c]] for c in start}
    return start, start_r, conf, moves


def _adjacency(ig):
    adj = {c: set() for c in ig.comp_ids}
    for a, b in ig.msg.t().tolist():
        adj[ig.comp_ids[a]].add(ig.comp_ids[b])
        adj[ig.comp_ids[b]].add(ig.comp_ids[a])
    return adj


class RandomPolicy:
    """control of the learned picks: as many components drawn at random, no structure"""

    def __init__(self, ig):
        self.ig = ig

    def __call__(self, rng, free_share, design=None):
        return set(rng.sample(self.ig.comp_ids, max(1, int(round(free_share * len(self.ig.comp_ids))))))


class CellPolicy:
    """frees whole cells: cells of the current design in random order until free_share of the components are free,
    then up to `partners` x that many communication partners of the freed components elsewhere. A freed cell can
    empty completely, so partitions (the bulk of the cost) can disappear or merge; the freed components may go to
    any cell."""

    def __init__(self, ig, partners=0.5):
        self.ig, self.adj, self.partners = ig, _adjacency(ig), float(partners)

    def __call__(self, rng, free_share, design=None):
        ids = self.ig.comp_ids
        k = max(1, int(round(free_share * len(ids))))
        if not design:
            return set(rng.sample(ids, k))
        by_cell = {}
        for c, j in design['x'].items():
            by_cell.setdefault(j, []).append(c)
        cells = sorted(by_cell)
        rng.shuffle(cells)
        free = set()
        for j in cells:   # whole cells; a cell beyond twice the target (few, crowded cells) only in part
            if len(free) >= k:
                break
            room = 2 * k - len(free)
            free.update(by_cell[j] if len(by_cell[j]) <= room else rng.sample(by_cell[j], room))
        nb = sorted({d for c in free for d in self.adj[c]} - free)
        rng.shuffle(nb)
        free.update(nb[:int(round(self.partners * k))])
        return free


class ClusterPolicy:
    """frees a cluster of the communication graph: breadth-first from a random component over the messages until
    free_share of the components are free (related removal, Shaw 1998; the graph-partition neighbourhoods of
    Light-MILPopt have the same idea); a new random seed when the cluster is exhausted"""

    def __init__(self, ig):
        self.ig, self.adj = ig, _adjacency(ig)

    def __call__(self, rng, free_share, design=None):
        ids = self.ig.comp_ids
        k = max(1, int(round(free_share * len(ids))))
        free, queue = set(), []
        while len(free) < k:
            if not queue:
                seed = rng.choice([c for c in ids if c not in free])
                free.add(seed)
                queue.append(seed)
                continue
            c = queue.pop(0)
            nb = sorted(self.adj[c] - free)
            rng.shuffle(nb)
            for d in nb:
                if len(free) < k:
                    free.add(d)
                    queue.append(d)
        return free


def make_guide(scfg, config_dir, architecture_json, num_locs):
    """guide(rho) for optimize_growth_sweep (configs/solver.json learn.search)"""
    model, ck = load_policy(scfg['model'])
    mode = scfg.get('mode', 'rl')
    k, kr = int(scfg.get('top_k', 3)), int(scfg.get('top_k_standby', 3))
    lns = {key: scfg[key] for key in ('time_share', 'init_s', 'iter_s', 'free_share', 'seed', 'max_iters', 'stop_gap', 'final_s') if key in scfg}
    fix_start = bool(scfg.get('fix_start', True))
    print(f"[Guide] policy {scfg['model']} (epoch {ck.get('epoch')}, round {ck.get('round')}), mode {mode}, "
          f"start {'fixed placement' if fix_start else 'hint'}")

    def guide(rho):
        t0 = time.time()
        ig = InstanceGraph(load_instance(config_dir, architecture_json, rho, num_locs))
        px, pr = predict(model, ig.data())
        start, start_r, allowed, allowed_r, conf = {}, {}, {}, {}, {}
        for cid, p in px.items():
            order = sorted(p, key=p.get, reverse=True)
            start[cid], allowed[cid], conf[cid] = order[0], order[:k], p[order[0]]
            order_r = sorted((j for j in pr[cid] if j != order[0]), key=pr[cid].get, reverse=True)
            allowed_r[cid] = order_r[:kr]
            if order_r:
                start_r[cid] = order_r[0]
        if fix_start:   # method 3 part 1: the (repaired) prediction is kept, the exact model fills in the rest
            start, start_r, conf, moves = repair_placement(ig, px, pr)
            print(f"[Guide] rho {rho:g}: placement predicted and repaired ({moves} components moved, "
                  f"{len(set(start.values()))} cells used) in {time.time() - t0:.1f} s")
        else:
            print(f"[Guide] rho {rho:g}: start design predicted in {time.time() - t0:.1f} s")
        g = {'start': start, 'start_standby': start_r, 'confidence': conf, 'lns': lns, 'fix_start': fix_start,
             'fix_standby': bool(scfg.get('fix_standby', False)),   # standbys: fixed with the placement or Gurobi's choice
             'start_rounds': int(scfg.get('start_rounds', 3)), 'start_free': float(scfg.get('start_free', 0.25))}
        if mode == 'predict':
            g.update({'allowed': allowed, 'allowed_standby': allowed_r})
        elif mode == 'random':
            g['policy'] = RandomPolicy(ig)
        elif mode == 'cells':
            g['policy'] = CellPolicy(ig, scfg.get('partners', 0.5))
        elif mode == 'cluster':
            g['policy'] = ClusterPolicy(ig)
        else:
            # progress feature on the horizon the policy was trained with (learn.rl.max_iters)
            g['policy'] = RLPolicy(model, ig, ck.get('horizon', 20), seed=int(scfg.get('seed', 1)),
                                   greedy=bool(scfg.get('greedy', False)))
        return g
    return guide
