"""
Policy of the learned search (HANDOFF 16.25): GPS layers (Rampasek et al., NeurIPS 2022) whose local part is an HGT
layer (Hu et al., WWW 2020) over the heterogeneous instance graph of learn/features.py, plus global attention over all
nodes of a graph, so a component sees every other component directly, not only through its neighbours.

HGT layer: node-type-specific key / query / value projections, relation-specific attention and message matrices with a
learned prior per head, one softmax over all incoming edges of a node across relations, a type-specific output
projection and a gated skip connection. The edge features of our graph (traffic, cable lengths, chain budgets, current
placement) enter every head's attention score as a bias and the message as an extra term; PyG's HGTConv takes none.

Heads: placement (score per comp-cell edge; softmax over the cells of a component), standby placement, destroy (logit
per component: free it in the next iteration of the search) and value (expected further cost reduction of the state).
"""
import math

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax, to_dense_batch

CAND = ('comp', 'cand', 'cell')


def _key(rel):
    return '__'.join(rel)


class HGTEdgeConv(nn.Module):
    def __init__(self, hidden, node_types, edge_dims, heads):
        super().__init__()
        self.h, self.d = heads, hidden // heads
        self.k = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.q = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.v = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.out = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.skip = nn.ParameterDict({t: nn.Parameter(torch.ones(1)) for t in node_types})
        self.w_att = nn.ParameterDict({_key(r): nn.Parameter(torch.eye(self.d).repeat(heads, 1, 1)) for r in edge_dims})
        self.w_msg = nn.ParameterDict({_key(r): nn.Parameter(torch.eye(self.d).repeat(heads, 1, 1)) for r in edge_dims})
        self.prior = nn.ParameterDict({_key(r): nn.Parameter(torch.ones(heads)) for r in edge_dims})
        self.e_att = nn.ModuleDict({_key(r): nn.Linear(ed, heads) for r, ed in edge_dims.items()})
        self.e_msg = nn.ModuleDict({_key(r): nn.Linear(ed, hidden) for r, ed in edge_dims.items()})
        self.rels = list(edge_dims)

    def forward(self, h, ei, ea):
        H, d = self.h, self.d
        K = {t: self.k[t](x).view(-1, H, d) for t, x in h.items()}
        Q = {t: self.q[t](x).view(-1, H, d) for t, x in h.items()}
        V = {t: self.v[t](x).view(-1, H, d) for t, x in h.items()}
        score, msg, idx = {}, {}, {}
        for rel in self.rels:
            s, _, t = rel
            src, dst = ei[rel]
            if src.numel() == 0:
                continue
            k = torch.einsum('ehd,hdf->ehf', K[s][src], self.w_att[_key(rel)])
            sc = (k * Q[t][dst]).sum(-1) * self.prior[_key(rel)] / math.sqrt(d) + self.e_att[_key(rel)](ea[rel])
            mg = torch.einsum('ehd,hdf->ehf', V[s][src], self.w_msg[_key(rel)]) + self.e_msg[_key(rel)](ea[rel]).view(-1, H, d)
            score.setdefault(t, []).append(sc)
            msg.setdefault(t, []).append(mg)
            idx.setdefault(t, []).append(dst)
        new = {}
        for t, x in h.items():
            if t not in score:
                new[t] = x
                continue
            sc, mg, ix = torch.cat(score[t]), torch.cat(msg[t]), torch.cat(idx[t])
            alpha = softmax(sc, ix, num_nodes=x.shape[0])
            agg = scatter(alpha.unsqueeze(-1) * mg, ix, dim=0, dim_size=x.shape[0], reduce='sum').view(-1, H * d)
            gate = torch.sigmoid(self.skip[t])
            new[t] = gate * self.out[t](nn.functional.gelu(agg)) + (1 - gate) * x
        return new


class GPSLayer(nn.Module):
    """local HGT layer + global multi-head attention over all nodes of each graph, summed, then a feed-forward block"""

    def __init__(self, hidden, node_types, edge_dims, heads, dropout):
        super().__init__()
        self.local = HGTEdgeConv(hidden, node_types, edge_dims, heads)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types})
        self.norm2 = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types})
        self.ffn = nn.ModuleDict({t: nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(2 * hidden, hidden)) for t in node_types})

    def forward(self, h, ei, ea, batch):
        loc = self.local(h, ei, ea)
        types = [t for t in h if h[t].shape[0] > 0]
        x = torch.cat([h[t] for t in types])
        b = torch.cat([batch[t] for t in types])
        order = torch.argsort(b, stable=True)
        dense, mask = to_dense_batch(x[order], b[order])
        g, _ = self.attn(dense, dense, dense, key_padding_mask=~mask, need_weights=False)
        glob = torch.empty_like(x)
        glob[order] = g[mask]
        out, start = {}, 0
        for t in h:
            if h[t].shape[0] == 0:
                out[t] = h[t]
                continue
            n = h[t].shape[0]
            z = self.norm1[t](loc[t] + h[t] + self.drop(glob[start:start + n]))
            out[t] = self.norm2[t](z + self.drop(self.ffn[t](z)))
            start += n
        return out


class PolicyNet(nn.Module):
    def __init__(self, node_dims, edge_dims, hidden=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        edge_dims = {tuple(r) if not isinstance(r, tuple) else r: int(v) for r, v in
                     (edge_dims.items() if isinstance(edge_dims, dict) else edge_dims)}
        self.cfg = {'node_dims': dict(node_dims), 'edge_dims': [[list(r), v] for r, v in edge_dims.items()],
                    'hidden': hidden, 'layers': layers, 'heads': heads, 'dropout': dropout}
        self.edge_dims = edge_dims
        self.enc = nn.ModuleDict({t: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden))
                                  for t, d in node_dims.items()})
        self.layers = nn.ModuleList(GPSLayer(hidden, list(node_dims), edge_dims, heads, dropout) for _ in range(layers))
        ce = edge_dims[CAND]
        self.head_x = nn.Sequential(nn.Linear(2 * hidden + ce, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_r = nn.Sequential(nn.Linear(2 * hidden + ce, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_free = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_v = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg['node_dims'], [(tuple(r), v) for r, v in cfg['edge_dims']], cfg['hidden'], cfg['layers'], cfg['heads'], cfg['dropout'])

    def forward(self, data):
        h = {t: self.enc[t](data[t].x) for t in self.enc}
        batch = {t: (data[t].batch if getattr(data[t], 'batch', None) is not None
                     else torch.zeros(data[t].num_nodes, dtype=torch.long, device=data[t].x.device)) for t in self.enc}
        ei = {rel: data[rel].edge_index for rel in self.edge_dims}
        ea = {rel: data[rel].edge_attr for rel in self.edge_dims}
        for layer in self.layers:
            h = layer(h, ei, ea, batch)
        src, dst = data[CAND].edge_index
        z = torch.cat([h['comp'][src], h['cell'][dst], data[CAND].edge_attr], dim=-1)
        n_graphs = int(batch['comp'].max()) + 1 if batch['comp'].numel() else 1
        pooled = torch.cat([scatter(h['comp'], batch['comp'], dim=0, dim_size=n_graphs, reduce='mean'),
                            scatter(h['cell'], batch['cell'], dim=0, dim_size=n_graphs, reduce='mean')], dim=-1)
        return {'x': self.head_x(z).squeeze(-1), 'r': self.head_r(z).squeeze(-1),
                'free': self.head_free(h['comp']).squeeze(-1), 'value': self.head_v(pooled).squeeze(-1)}


def edge_dims_of(data):
    return {rel: int(data[rel].edge_attr.shape[1]) for rel in data.edge_types}


def node_dims_of(data):
    return {t: int(data[t].x.shape[1]) for t in data.node_types}


def _label_edges(data, y):
    src, dst = data[CAND].edge_index
    offset = data['cell'].ptr[data['comp'].batch] if getattr(data['comp'], 'batch', None) is not None else torch.zeros_like(y)
    y_glob = torch.where(y >= 0, y + offset, torch.full_like(y, -1))
    return (dst == y_glob[src]) & (y[src] >= 0), y_glob


def placement_losses(data, out):
    """cross entropy of the primary cell and of the standby cell (softmax over the cells other than the primary's);
    top-1 / top-3 hit counts of the primary cell"""
    lx, lr = out['x'], out['r']
    src, dst = data[CAND].edge_index
    is_x, y_glob = _label_edges(data, data['comp'].y_cell)
    loss_x = -torch.log(softmax(lx, src) + 1e-12)[is_x].mean()
    is_r, _ = _label_edges(data, data['comp'].y_standby)
    lp_r = torch.log(softmax(lr.masked_fill(dst == y_glob[src], float('-inf')), src) + 1e-12)
    loss_r = -lp_r[is_r].mean() if bool(is_r.any()) else lx.new_zeros(())
    tgt = scatter(torch.where(is_x, lx, torch.full_like(lx, float('-inf'))), src, dim=0, dim_size=data['comp'].num_nodes, reduce='max')
    rank = scatter((lx > tgt[src]).float(), src, dim=0, dim_size=data['comp'].num_nodes, reduce='sum')
    has = data['comp'].y_cell >= 0
    return loss_x, loss_r, {'n': int(has.sum()), 'top1': int(((rank == 0) & has).sum()), 'top3': int(((rank < 3) & has).sum())}


def moved_loss(data, out):
    """destroy head as a detector of misplaced components: binary cross entropy against y_moved (positives weighted
    by their rarity)"""
    y = data['comp'].y_moved
    pos = y.sum().clamp(min=1.0)
    w = torch.where(y > 0, (y.numel() - pos) / pos, torch.ones_like(y))
    return nn.functional.binary_cross_entropy_with_logits(out['free'], y, weight=w)


def sample_free(logits, k, generator=None):
    """k components without replacement by Gumbel top-k (= sequential sampling from the softmax); returns the indices
    in selection order"""
    u = torch.rand(logits.shape, generator=generator).clamp(1e-9, 1 - 1e-9)
    return torch.topk(logits - torch.log(-torch.log(u)), k).indices


def plackett_luce_logp(logits, order):
    """log probability of choosing `order` (indices, in sequence) without replacement from softmax(logits):
    sum over the steps of logit(chosen) - log(sum of exp over the not yet chosen), vectorised"""
    top = logits.max().detach()
    e = torch.exp(logits - top)
    taken = torch.cumsum(e[order], dim=0) - e[order]          # mass removed before each step
    denom = (e.sum() - taken).clamp(min=1e-12)
    return (logits[order] - top - torch.log(denom)).sum()


def entropy_free(logits):
    p = torch.softmax(logits, dim=0)
    return -(p * torch.log(p + 1e-12)).sum()


@torch.no_grad()
def predict(model, data):
    """placement probabilities per component over the cells: {comp_id: {cell_id: p}}, primary and standby"""
    model.eval()
    out = model(data)
    src, dst = data[CAND].edge_index
    px, pr = softmax(out['x'], src), softmax(out['r'], src)
    out_x, out_r = {}, {}
    for e in range(src.numel()):
        c, j = data.comp_ids[int(src[e])], data.cell_ids[int(dst[e])]
        out_x.setdefault(c, {})[j] = float(px[e])
        out_r.setdefault(c, {})[j] = float(pr[e])
    return out_x, out_r
