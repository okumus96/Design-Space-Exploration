"""
Training instances of the learned search (HANDOFF 16.25): variants of a base instance with a random subset of the
candidate cells, a random robustness level rho, random chains and other components left out, and perturbed demands and
message sizes. Every variant is a self-contained run directory, cfg/ (the config files the optimiser and the checker
read) plus arch.json, so it is solved and checked like any other run. Settings: configs/solver.json, learn.instances.
"""
import json
import os
import random
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_FILES = ('vehicle.json', 'sensors.json', 'actuators.json', 'buses.json', 'partitions.json', 'hardwares.json',
             'uncertainty.json', 'sites.json', 'distances.json', 'chains.json', 'solver.json')
DRIVER_RE = re.compile(r'^SC_\d+_Driver_(.+)$')


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def make_variant(seed, out_dir, spec):
    """Write variant `seed` of the base instance into out_dir (cfg/, arch.json, variant.json) and return its
    description. The same seed always gives the same variant."""
    rng = random.Random(seed)
    base_cfg = os.path.join(REPO, spec.get('base_config_dir', 'configs'))
    arch = _load(os.path.join(REPO, spec['base_architecture']))
    chains = _load(os.path.join(base_cfg, 'chains.json'))
    sites = _load(os.path.join(base_cfg, 'sites.json'))
    unc = _load(os.path.join(base_cfg, 'uncertainty.json'))
    sens = _load(os.path.join(base_cfg, 'sensors.json'))
    acts = _load(os.path.join(base_cfg, 'actuators.json'))

    # robustness level and candidate cells (kept in the order of sites.json)
    rho = 0.0 if rng.random() < float(spec.get('rho_zero_share', 0.15)) else round(rng.uniform(0.05, 1.0), 2)
    lo, hi = spec.get('cells', [4, 10])
    keep_sites = {s['id'] for s in rng.sample(sites['sites'], rng.randint(int(lo), int(hi)))}
    sites['sites'] = [s for s in sites['sites'] if s['id'] in keep_sites]

    # structure: whole chains and single components outside the kept chains are left out with the same share q
    in_any = {s for c in chains['chains'] for s in c['swcs']}
    q_lo, q_hi = spec.get('removal_share', [0.0, 0.4])
    while True:
        q = rng.uniform(float(q_lo), float(q_hi))
        kept_chains = [c for c in chains['chains'] if rng.random() >= q]
        in_kept = {s for c in kept_chains for s in c['swcs']}
        keep = {c['id'] for c in arch['software_components']
                if c['id'] in in_kept or (c['id'] not in in_any and rng.random() >= q)}
        if len(keep) >= int(spec.get('min_components', 15)):
            break
    f_lo, f_hi = (float(v) for v in spec.get('demand_factor', [0.8, 1.2]))
    comps = []
    for c in arch['software_components']:
        if c['id'] not in keep:
            continue
        c = dict(c)
        c['receives_from'] = [s for s in c.get('receives_from', []) if s in keep]
        if c.get('redundant_with') not in keep:
            c['redundant_with'] = None
        for k in ('cpu_req', 'ram_req', 'rom_req'):
            if c.get(k):
                c[k] = max(1, int(round(c[k] * rng.uniform(f_lo, f_hi))))
        if c.get('tops_req'):
            c['tops_req'] = round(float(c['tops_req']) * rng.uniform(f_lo, f_hi), 4)
        if c.get('msg_kbit') is not None:
            c['msg_kbit'] = round(float(c['msg_kbit']) * rng.uniform(f_lo, f_hi), 4)
        comps.append(c)
    arch['software_components'] = comps
    chains['chains'] = kept_chains
    # devices whose driver component is left out leave the vehicle with it
    drivers = {m.group(1) for c in comps for m in [DRIVER_RE.match(c['id'])] if m and c.get('type') in ('sensor', 'actuator')}
    sens['sensors'] = [d for d in sens['sensors'] if d['id'] in drivers]
    acts['actuators'] = [d for d in acts['actuators'] if d['id'] in drivers]
    unc['rho']['levels'] = [rho]

    cfg = os.path.join(out_dir, 'cfg')
    os.makedirs(cfg, exist_ok=True)
    for f in CFG_FILES:
        shutil.copy(os.path.join(base_cfg, f), os.path.join(cfg, f))
    for obj, f in ((sites, 'sites.json'), (chains, 'chains.json'), (unc, 'uncertainty.json'),
                   (sens, 'sensors.json'), (acts, 'actuators.json')):
        _dump(obj, os.path.join(cfg, f))
    _dump(arch, os.path.join(out_dir, 'arch.json'))
    desc = {'seed': seed, 'rho': rho, 'cells': [s['id'] for s in sites['sites']], 'components': len(comps),
            'chains': len(kept_chains), 'removal_share': round(q, 3), 'devices': len(sens['sensors']) + len(acts['actuators'])}
    _dump(desc, os.path.join(out_dir, 'variant.json'))
    return desc
