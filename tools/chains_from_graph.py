#!/usr/bin/env python3
"""
Derive the cause-effect chains of the architecture from its data-flow graph.

  dse/bin/python tools/chains_from_graph.py [--architecture configs/full_architecture_v2.json] [--out configs/chains.json]

The graph is the 'receives_from' relation of the software components: an edge src -> dst means dst consumes data of
src. Sources are the sensor-driver components, sinks the actuator-driver components. For every actuator the script
takes the longest simple path from any sensor driver (the chain with the most hops, ties broken by component id) and,
for the inertial/positioning sensors, the shortest path to each actuator of the vehicle-motion group. Every chain gets
a network budget in milliseconds: the end-to-end bound of the chain minus the execution times of its components,
i.e. the time the data may spend on cables and switches (see HANDOFF section 14.3 for the model).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# network latency budget per actuator (ms): what the data may spend on the in-vehicle network along the whole chain
NET_BUDGET_MS = {'BRAKE': 20.0, 'STEERING': 20.0, 'MOTOR': 30.0, 'LIGHT': 200.0, 'HVAC': 500.0}
MOTION_ACTUATORS = ('BRAKE', 'STEERING', 'MOTOR')
POSITIONING_SENSORS = ('IMU', 'GPS')


def actuator_type(sc_id):
    for t in NET_BUDGET_MS:
        if t in sc_id.upper():
            return t
    return None


def simple_paths(succ, start, is_sink):
    """All simple paths from start to any sink (DFS)."""
    out = []

    def dfs(node, path, seen):
        if is_sink(node) and len(path) > 1:
            out.append(list(path))
            return
        for nxt in succ.get(node, []):
            if nxt not in seen:
                seen.add(nxt); path.append(nxt)
                dfs(nxt, path, seen)
                path.pop(); seen.discard(nxt)

    dfs(start, [start], {start})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--architecture', default='configs/full_architecture_v2.json')
    ap.add_argument('--out', default='configs/chains.json')
    args = ap.parse_args()
    scs = json.load(open(args.architecture))['software_components']
    by_id = {s['id']: s for s in scs}
    succ = defaultdict(list)
    for s in scs:
        for src in s.get('receives_from', []):
            if src in by_id:
                succ[src].append(s['id'])
    sources = [s['id'] for s in scs if s['type'] == 'sensor']
    sinks = {s['id'] for s in scs if s['type'] == 'actuator'}

    paths_to = defaultdict(list)
    for s0 in sources:
        for p in simple_paths(succ, s0, lambda n: n in sinks):
            paths_to[p[-1]].append(p)

    chains = []
    for act in sorted(sinks):
        cands = paths_to.get(act, [])
        if not cands:
            print(f"WARNING: no sensor reaches {act}", file=sys.stderr)
            continue
        longest = max(cands, key=lambda p: (len(p), [-ord(c) for c in ''.join(p)]))
        longest = max([p for p in cands if len(p) == len(longest)], key=lambda p: ''.join(p))
        chains.append({'id': f"chain_{act}", 'actuator': act, 'sensor': longest[0], 'swcs': longest, 'hops': len(longest) - 1,
                       'net_budget_ms': NET_BUDGET_MS[actuator_type(act)], 'rule': 'longest sensor-to-actuator path'})
        if actuator_type(act) in MOTION_ACTUATORS:
            fast = [p for p in cands if any(k in p[0].upper() for k in POSITIONING_SENSORS)]
            if fast:
                shortest = min(fast, key=lambda p: (len(p), ''.join(p)))
                chains.append({'id': f"chain_{act}_positioning", 'actuator': act, 'sensor': shortest[0], 'swcs': shortest,
                               'hops': len(shortest) - 1, 'net_budget_ms': NET_BUDGET_MS[actuator_type(act)] / 2,
                               'rule': 'shortest path from a positioning sensor'})
    out = {'_generated_by': 'tools/chains_from_graph.py (receives_from graph of the architecture)',
           '_comment': 'Generated file, do not edit; change the architecture or the budgets in the script and re-run. '
                       'net_budget_ms = time the data of the chain may spend on the network (cables, switches, synchronisation).',
           'architecture': os.path.basename(args.architecture), 'chains': chains}
    json.dump(out, open(args.out, 'w'), indent=2)
    print(f"{len(chains)} chains written to {args.out}")
    for c in chains:
        print(f"  {c['id']:38s} {c['hops']:2d} hops  budget {c['net_budget_ms']:6.1f} ms  {c['sensor']} -> ... -> {c['actuator']}")


if __name__ == '__main__':
    main()
