"""Summary of a schema-v3 architecture (configs/arch_v3): components per domain, ASIL and period mix, CPU / memory
totals, link sizes and rates, chain lengths and bounds, measured vs generated shares. Read-only."""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'arch_v3')
comps = json.load(open(os.path.join(d, 'swc.json')))['components']
flows = json.load(open(os.path.join(d, 'flows.json')))['flows']
chains = json.load(open(os.path.join(d, 'chains.json')))['chains']


def pct(c):
    t = sum(c.values()) or 1
    return {k: f"{v / t:.0%}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


aw = [c for c in comps if c['id'].startswith('AW:')]
gen = [c for c in comps if not c['id'].startswith('AW:')]
print(f"components {len(comps)}: Autoware (measured) {len(aw)}, non-AD (generated from standards + distributions) {len(gen)}")
print('per domain:', dict(Counter(c['domain'] for c in comps)))
print('non-AD ASIL mix:', pct(Counter(c['asil'] for c in gen)))
print('non-AD period mix (ms):', pct(Counter(c['period_ms'] for c in gen)))
print(f"non-AD CPU {sum(c['cpu_dmips'] for c in gen):,.0f} DMIPS, ROM {sum(c['rom_kb'] for c in gen) / 1024:,.0f} MB, RAM {sum(c['ram_kb'] for c in gen) / 1024:,.0f} MB")
cpu = [c['cpu_mean_pct'] for c in aw if c.get('cpu_mean_pct') is not None]
rss = [c['ram_kb'] / 1024.0 for c in aw if c.get('ram_kb') is not None]
print(f"Autoware CPU measured for {len(cpu)}/{len(aw)} nodes: {sum(cpu):.0f} % of one core; RSS {sum(rss):,.0f} MB; processing times for {sum(1 for c in aw if c.get('processing_time_ms'))} nodes")
print(f"links {len(flows)}: ", dict(Counter(f['kind'] for f in flows)))
b = [f['bytes'] for f in flows if f.get('bytes')]
hz = [f['rate_hz'] for f in flows if f.get('rate_hz')]
print(f"  bytes known {len(b)}: median {statistics.median(b):.0f}, p90 {sorted(b)[int(0.9 * (len(b) - 1))]:.0f}, max {max(b):,}; rate known {len(hz)}: median {statistics.median(hz):.0f} Hz")
tot_bw = sum(f['bytes'] * f['rate_hz'] * 8 / 1e6 for f in flows if f.get('bytes') and f.get('rate_hz'))
print(f"  total traffic of the links with known size and rate: {tot_bw:,.1f} Mbit/s")
L = [len(c['swcs']) for c in chains]
bd = [c['e2e_bound_ms'] for c in chains if c.get('e2e_bound_ms')]
print(f"chains {len(chains)}: length median {statistics.median(L):.0f} (max {max(L)}); bound median {statistics.median(bd):.0f} ms (min {min(bd):.0f}, max {max(bd):.0f}); FTTI known for {sum(1 for c in chains if c.get('ftti_ms'))}")
src = Counter(('measured' if c['source'].startswith('measured') else 'generated') for c in comps)
print('component sources:', dict(src))
