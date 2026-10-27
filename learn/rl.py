#!/usr/bin/env python3
"""
Reinforcement learning of the destroy policy (HANDOFF 16.25): PPO on episodes of the exact search. An episode is one
training variant: a short exact solve from the policy's start design (init_s), then max_iters iterations in which the
policy frees free_share of the components and the exact model re-solves them for iter_s seconds with the rest fixed;
the reward of an iteration is the cost reduction in % of the start cost. Every design comes out of the MILP with its
lazy constraints, so the policy can only change how fast good designs are found, never their validity.

  dse/bin/python learn/rl.py train <out_dir> <start_policy.pt or random>  rounds of episodes + PPO updates (learn.rl)
  dse/bin/python learn/rl.py episode <variant_dir> <policy.pt> <seed> <out.pt>   one episode (run by train)

Episodes run as separate processes, at most max_sessions at once (each holds one Gurobi WLS session); the PPO update
runs on the GPU when there is one. Settings: configs/solver.json, learn.rl.
"""
import concurrent.futures
import glob
import json
import os
import random
import subprocess
import sys
import time

import torch
from torch_geometric.data import Batch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, 'dse', 'bin', 'python')
sys.path.insert(0, REPO)
from learn.features import InstanceGraph, load_instance              # noqa: E402
from learn.gen_data import learn_cfg                                  # noqa: E402
from learn.model import entropy_free, plackett_luce_logp, predict     # noqa: E402
from learn.search import RLPolicy, load_policy                        # noqa: E402


def variant_graph(vdir):
    with open(os.path.join(vdir, 'variant.json')) as f:
        rho = json.load(f)['rho']
    return InstanceGraph(load_instance(os.path.join(vdir, 'cfg'), os.path.join(vdir, 'arch.json'), rho))


def episode(vdir, policy_path, seed, out):
    """one episode on a training variant; writes the steps (state, action, log probability, value, reward)"""
    from config_reader import ConfigReader
    from architecture_loader import load_architecture_from_json
    from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS
    torch.set_num_threads(2)
    cfg = learn_cfg().get('rl', {})
    model, ck = load_policy(policy_path)
    ig = variant_graph(vdir)
    policy = RLPolicy(model, ig, cfg.get('max_iters', 15), seed=seed)
    px, pr = predict(model, ig.data())
    start = {c: max(p, key=p.get) for c, p in px.items()}
    start_r = {c: max((j for j in pr[c] if j != start[c]), key=pr[c].get) for c in pr if len(pr[c]) > 1}
    lns = {'init_s': cfg.get('init_s', 45), 'iter_s': cfg.get('iter_s', 10), 'free_share': cfg.get('free_share', 0.2),
           'max_iters': cfg.get('max_iters', 15), 'seed': seed, 'stop_gap': 0.0, 'final_s': cfg.get('final_s', 5)}
    guide = {'policy': policy, 'start': start, 'start_standby': start_r, 'lns': lns}
    cr = ConfigReader(os.path.join(vdir, 'cfg'))
    partitions = cr.get_partitions()
    partitions['hw_capacity'] = cr.get_hardware_capacities()
    unc = {k: True for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    scs, comm, sensors, actuators, cable_types, locations = load_architecture_from_json(os.path.join(vdir, 'arch.json'), config_reader=cr)
    params = ','.join(p for p in (cfg.get('gurobi_params', ''), f"Threads={int(cfg.get('threads', 8))}", f"Seed={seed % 1000}") if p)
    tl = int(lns["init_s"] + lns["max_iters"] * (lns["iter_s"] + 2) + 20)   # slack for model updates between iterations
    t0 = time.time()
    sols = AssignmentOptimizer().optimize_growth_sweep(
        scs, locations, sensors, actuators, cable_types, comm, partitions=partitions, hardwares=cr.get_hardwares(),
        interfaces=cr.get_interfaces(), backbone=cr.get_backbone_config(), phase2_time=1, gurobi_params=params,
        lns=None, chains=cr.get_chains(), solve_time_limit=tl, uncertainty=unc, guide=lambda rho: guide)
    torch.save({'variant': vdir, 'seed': seed, 'steps': policy.steps, 'start_cost': policy.start_cost,
                'final_cost': float(sols[0]['total_cost']) if sols else None, 'seconds': time.time() - t0}, out)


def advantages(steps, gamma, lam):
    """GAE over the rewarded steps of one episode; returns (advantages, returns)"""
    adv, last = [0.0] * len(steps), 0.0
    for t in reversed(range(len(steps))):
        nv = steps[t + 1]['value'] if t + 1 < len(steps) else 0.0
        delta = steps[t]['reward'] + gamma * nv - steps[t]['value']
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, [a + s['value'] for a, s in zip(adv, steps)]


def ppo_update(model, episodes, graphs, opt, cfg, dev):
    p = cfg.get('ppo', {})
    gamma, lam, clip = float(p.get('gamma', 0.95)), float(p.get('lam', 0.9)), float(p.get('clip', 0.2))
    trans = []
    for ep in episodes:
        steps = [s for s in ep['steps'] if 'reward' in s]
        if steps:
            adv, ret = advantages(steps, gamma, lam)
            trans += [(graphs[ep['variant']], s, a, r) for s, a, r in zip(steps, adv, ret)]
    if not trans:
        return {}
    a = torch.tensor([t[2] for t in trans])
    a = (a - a.mean()) / (a.std() + 1e-8) if len(trans) > 1 else a
    trans = [(g, s, float(ai), r) for (g, s, _, r), ai in zip(trans, a)]
    model.train()
    stats = {'pg': 0.0, 'v': 0.0, 'ent': 0.0, 'n': 0}
    mb = int(p.get('minibatch', 16))
    for _ in range(int(p.get('epochs', 4))):
        random.shuffle(trans)
        for i in range(0, len(trans), mb):
            chunk = trans[i:i + mb]
            batch = Batch.from_data_list([g.data(s['design'], s['freed_last'], s['it_frac']) for g, s, _, _ in chunk]).to(dev)
            out = model(batch)
            pg, vl, ent = [], [], []
            for k, (g, s, adv_k, ret_k) in enumerate(chunk):
                lg = out['free'][batch['comp'].batch == k]
                logp = plackett_luce_logp(lg, torch.tensor(s['order'], device=dev))
                ratio = torch.exp(logp - s['logp'])
                pg.append(-torch.min(ratio * adv_k, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_k))
                vl.append((out['value'][k] - ret_k) ** 2)
                ent.append(entropy_free(lg))
            loss = torch.stack(pg).mean() + float(p.get('value_coef', 0.5)) * torch.stack(vl).mean() \
                - float(p.get('entropy', 0.01)) * torch.stack(ent).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats['pg'] += float(torch.stack(pg).mean())
            stats['v'] += float(torch.stack(vl).mean())
            stats['ent'] += float(torch.stack(ent).mean())
            stats['n'] += 1
    return {k: v / max(1, stats['n']) for k, v in stats.items() if k != 'n'} | {'transitions': len(trans)}


def train(out_dir, start_policy):
    cfg = learn_cfg().get('rl', {})
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, 'episodes'), exist_ok=True)
    pool = sorted(d for pat in cfg.get('variant_dirs', ['results/learn/data_v1/v*'])
                  for d in glob.glob(os.path.join(REPO, pat)) if os.path.exists(os.path.join(d, 'variant.json')))
    print(f"{len(pool)} training variants", flush=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if start_policy == 'random':   # no pretraining: a freshly initialised policy (sizes of learn.train)
        from learn.model import PolicyNet, edge_dims_of, node_dims_of
        tc = learn_cfg().get('train', {})
        torch.manual_seed(int(cfg.get('seed', 0)))
        probe = variant_graph(pool[0]).data()
        model = PolicyNet(node_dims_of(probe), edge_dims_of(probe), int(tc.get('hidden', 128)), int(tc.get('layers', 3)),
                          int(tc.get('heads', 4)), float(tc.get('dropout', 0.1)))
    else:
        model, ck = load_policy(start_policy)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get('ppo', {}).get('lr', 3e-4)))
    latest = os.path.join(out_dir, 'policy_latest.pt')
    save = lambda rnd: torch.save({'state': {k: v.cpu() for k, v in model.state_dict().items()}, 'cfg': model.cfg,
                                   'round': rnd, 'horizon': int(cfg.get('max_iters', 15)), 'start_policy': start_policy}, latest)
    save(0)
    graphs, rng = {}, random.Random(int(cfg.get('seed', 0)))
    log = open(os.path.join(out_dir, 'train_log.jsonl'), 'a')
    for rnd in range(1, int(cfg.get('rounds', 100)) + 1):
        t0 = time.time()
        snap = os.path.join(out_dir, 'policy_round.pt')
        subprocess.run(['cp', latest, snap], check=True)
        jobs = [(rng.choice(pool), rnd * 1000 + k) for k in range(int(cfg.get('episodes_per_round', 16)))]

        def run(job):
            vdir, seed = job
            out = os.path.join(out_dir, 'episodes', f"r{rnd:04d}_{seed}.pt")
            with open(out[:-3] + '.log', 'w') as f:
                subprocess.run([PY, os.path.abspath(__file__), 'episode', vdir, snap, str(seed), out], stdout=f, stderr=subprocess.STDOUT)
            return out

        sessions = int(learn_cfg().get('rl', {}).get('max_sessions', 8))   # re-read: can be raised while training runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=sessions) as ex:
            files = list(ex.map(run, jobs))
        episodes = [torch.load(f, weights_only=False) for f in files if os.path.exists(f)]
        for ep in episodes:
            if ep['variant'] not in graphs:
                graphs[ep['variant']] = variant_graph(ep['variant'])
        st = ppo_update(model, episodes, graphs, opt, cfg, dev)
        save(rnd)
        subprocess.run(['cp', latest, os.path.join(out_dir, f"policy_r{rnd:04d}.pt")], check=True)
        gains = [100 * (e['start_cost'] - e['final_cost']) / e['start_cost'] for e in episodes
                 if e.get('start_cost') and e.get('final_cost')]
        row = {'round': rnd, 'episodes': len(episodes), 'with_steps': len(gains),
               'mean_gain_pct': sum(gains) / len(gains) if gains else None,
               'mean_step_reward': (sum(s.get('reward', 0) for e in episodes for s in e['steps']) /
                                    max(1, sum(len(e['steps']) for e in episodes))), 'seconds': time.time() - t0, **st}
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'train' and len(sys.argv) == 4:
        train(sys.argv[2], sys.argv[3])
    elif cmd == 'episode' and len(sys.argv) == 6:
        episode(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        print(__doc__)
