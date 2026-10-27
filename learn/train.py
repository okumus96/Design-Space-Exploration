#!/usr/bin/env python3
"""
Pretraining of the search policy (learn/model.py) on the checked designs of learn/gen_data.py (HANDOFF 16.25), before
the reinforcement learning (learn/rl.py).

  dse/bin/python learn/train.py <data_dir> <model_out.pt>

Two tasks on one encoder: placement (cross entropy of the cell and of the standby cell of the good design) and the
destroy head as a detector of misplaced components: the good design is perturbed (a random share of its components
moved to other cells) and the head learns which components sit elsewhere than in the good design, i.e. which ones an
iteration of the search should free. A share of the samples has no design (placement only, the start of a search).
Settings: configs/solver.json, learn.train. Variants are split into training and validation (val_share); the model of
the epoch with the lowest validation loss is kept. The instance graphs are cached in <data_dir>/graphs.pt.
"""
import glob
import json
import math
import os
import random
import sys
import time

import torch
from torch_geometric.loader import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from learn.features import load_variant                                                         # noqa: E402
from learn.gen_data import learn_cfg                                                            # noqa: E402
from learn.model import PolicyNet, edge_dims_of, moved_loss, node_dims_of, placement_losses     # noqa: E402


def load_graphs(data_dir):
    cache = os.path.join(data_dir, 'graphs.pt')
    dirs = sorted(d for d in glob.glob(os.path.join(data_dir, 'v*')) if os.path.exists(os.path.join(d, 'label.json')))
    names = [os.path.basename(d) for d in dirs]
    if os.path.exists(cache):
        c = torch.load(cache, weights_only=False)
        if c['variants'] == names:
            return c['items']
    items = [it for it in (load_variant(d) for d in dirs) if it is not None]
    torch.save({'variants': names, 'items': items}, cache)
    return items


def perturbed_state(ig, label, rng, cfg):
    """a state of the search near the good design: a share of the components moved to other cells (and some standbys),
    random recently freed components and episode progress; None as design with probability unperturbed_share"""
    if rng.random() < float(cfg.get('unperturbed_share', 0.3)):
        return None, None, 0.0
    lo, hi = cfg.get('perturb_share', [0.05, 0.4])
    x = dict(label['assignment'])
    r = {c: s['cell'] for c, s in (label.get('standby') or {}).items()}
    cells = ig.cell_ids
    comps = [c for c in ig.comp_ids if c in x]
    for c in rng.sample(comps, max(1, int(round(rng.uniform(float(lo), float(hi)) * len(comps))))):
        if len(cells) > 1:
            x[c] = rng.choice([j for j in cells if j != x[c]])
    for c in list(r):
        if rng.random() < 0.2 or r[c] == x.get(c):
            others = [j for j in cells if j != x.get(c)]
            if others:
                r[c] = rng.choice(others)
    freed = rng.sample(comps, min(len(comps), int(0.2 * len(comps)))) if rng.random() < 0.5 else []
    return {'x': x, 'r': r}, freed, rng.random()


def batches(items, rng, cfg, bs, shuffle):
    data = []
    for ig, label in items:
        design, freed, it = perturbed_state(ig, label, rng, cfg)
        data.append(ig.data(design, freed, it, label))
    return DataLoader(data, batch_size=bs, shuffle=shuffle)


def run_epoch(model, loader, dev, cfg, opt=None):
    w_r, w_m = float(cfg.get('standby_weight', 0.5)), float(cfg.get('moved_weight', 1.0))
    tot, n, hits, prec = 0.0, 0, {'n': 0, 'top1': 0, 'top3': 0}, [0, 0]
    model.train(opt is not None)
    for b in loader:
        b = b.to(dev)
        with torch.set_grad_enabled(opt is not None):
            out = model(b)
            lx, lr, st = placement_losses(b, out)
            lm = moved_loss(b, out)
            loss = lx + w_r * lr + w_m * lm
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * b.num_graphs
        n += b.num_graphs
        for k in hits:
            hits[k] += st[k]
        y = b['comp'].y_moved
        for g in range(b.num_graphs):   # precision of the destroy head: its top-m picks among the m moved components
            sel = b['comp'].batch == g
            m = int(y[sel].sum())
            if m:
                top = torch.topk(out['free'][sel], m).indices
                prec[0] += int(y[sel][top].sum())
                prec[1] += m
    return tot / max(1, n), hits, prec[0] / max(1, prec[1])


def main(data_dir, out):
    cfg = learn_cfg().get('train', {})
    items = load_graphs(data_dir)
    rng = random.Random(int(cfg.get('seed', 0)))
    rng.shuffle(items)
    n_val = max(1, int(len(items) * float(cfg.get('val_share', 0.1))))
    val, train = items[:n_val], items[n_val:]
    print(f"{len(items)} checked variants: {len(train)} training, {len(val)} validation", flush=True)
    torch.manual_seed(int(cfg.get('seed', 0)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = train[0][0].data()
    model = PolicyNet(node_dims_of(probe), edge_dims_of(probe), int(cfg.get('hidden', 128)), int(cfg.get('layers', 3)),
                      int(cfg.get('heads', 4)), float(cfg.get('dropout', 0.1))).to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 1e-3)), weight_decay=float(cfg.get('weight_decay', 1e-4)))
    bs = int(cfg.get('batch_size', 16))
    va_loader = batches(val, random.Random(12345), cfg, bs, False)   # fixed validation states
    best, best_epoch, t0 = math.inf, -1, time.time()
    for epoch in range(int(cfg.get('epochs', 300))):
        tr_loss, tr_hits, tr_prec = run_epoch(model, batches(train, rng, cfg, bs, True), dev, cfg, opt)
        va_loss, va_hits, va_prec = run_epoch(model, va_loader, dev, cfg)
        print(f"epoch {epoch:3d}  train {tr_loss:.4f} top1 {tr_hits['top1'] / max(1, tr_hits['n']):.3f} moved-prec {tr_prec:.3f}  "
              f"val {va_loss:.4f} top1 {va_hits['top1'] / max(1, va_hits['n']):.3f} top3 {va_hits['top3'] / max(1, va_hits['n']):.3f} "
              f"moved-prec {va_prec:.3f}  {time.time() - t0:.0f} s", flush=True)
        if va_loss < best:
            best, best_epoch = va_loss, epoch
            torch.save({'state': model.state_dict(), 'cfg': model.cfg, 'epoch': epoch, 'val_loss': va_loss,
                        'val_top1': va_hits['top1'] / max(1, va_hits['n']), 'val_top3': va_hits['top3'] / max(1, va_hits['n']),
                        'val_moved_precision': va_prec,
                        'train_variants': sorted(ig.variant for ig, _ in train), 'val_variants': sorted(ig.variant for ig, _ in val)}, out)
        elif epoch - best_epoch >= int(cfg.get('patience', 30)):
            break
    print(f"best epoch {best_epoch}, validation loss {best:.4f}; model saved to {out}")
    ck = torch.load(out, weights_only=False)
    with open(out + '.json', 'w') as f:
        json.dump({k: v for k, v in ck.items() if k != 'state'}, f, indent=1)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
