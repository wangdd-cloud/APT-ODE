"""
Ablation study for APT-ODE (RQ2).
- Main ablation (Table 4): Full model vs. w/o AP, w/o ODE, w/o SI
- Design-choice ablation (Table 5): JSD->SKL/WD, Mean->GRU/Transformer, Cosine->L2
"""
import sys, logging, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from apt_ode import (APTODE, RecDataset, PairwiseDataset, pad_collate,
                     bpr_loss, dyn_loss, run_eval, set_seed,
                     apt_partition, _make_increasing)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ======================================================================
# Part A: Main ablation variants (Table 4)
# ======================================================================

class FixedWindowPartitioner:
    """w/o AP: fixed-window segmentation."""
    def __init__(self, w=5):
        self.w = w

    def partition(self, embs, times):
        n = len(times)
        w = self.w
        bounds = [times[0]]
        segs = []
        start = 0
        for i in range(w, n, w):
            bounds.append(times[min(i, n - 1)])
            segs.append((start, min(i - 1, n - 1)))
            start = i
        if start < n:
            if times[-1] > bounds[-1]:
                bounds.append(times[-1])
            segs.append((start, n - 1))
        if not segs:
            segs = [(0, n - 1)]
            if len(bounds) < 2:
                bounds = [times[0], times[-1]]
        return bounds, segs


class GRUEncoder(nn.Module):
    """GRU-based discrete update replacing ODE solver (for w/o ODE variant)."""
    def __init__(self, d, h=128):
        super().__init__()
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.d = d

    def forward(self, z, env, times_len):
        steps = max(1, times_len)
        h0 = (z + env).unsqueeze(0).unsqueeze(0)
        dummy_input = env.unsqueeze(0).unsqueeze(0).repeat(1, steps, 1)
        out, _ = self.gru(dummy_input, h0)
        return out.squeeze(0)


def _evolve_with_partitioner(model, partitioner, uid, items, times, t_target,
                              reset_state_each_seg=False, use_gru=False,
                              gru_module=None):
    """Shared evolution logic for all ablation variants."""
    if items.dim() == 0:
        items = items.unsqueeze(0)

    dev = items.device
    encoded = model.enc(items)
    bounds, segs = partitioner(encoded.detach(), times)

    z = model.user_emb(uid)
    traj_z, traj_target = [], []

    for seg_idx, (start, end) in enumerate(segs):
        if start > end:
            continue

        env = encoded[start:end + 1].mean(0)

        if reset_state_each_seg and seg_idx > 0:
            z = encoded[start].detach()

        tau_start = bounds[seg_idx]
        tau_end = bounds[seg_idx + 1] if seg_idx + 1 < len(bounds) else bounds[-1]

        seg_interaction_times = times[start:end + 1]
        all_times = set()
        all_times.add(tau_start)
        all_times.add(tau_end)
        for t in seg_interaction_times:
            if tau_start <= t <= tau_end:
                all_times.add(t)
        all_times = sorted(all_times)

        if len(all_times) < 2:
            traj_z.append(z)
            traj_target.append(encoded[start])
            continue

        if use_gru and gru_module is not None:
            dt = tau_end - tau_start
            steps = max(1, int(dt * 100))
            zt = gru_module(z, env, steps)
            z = zt[-1]
            traj_z.append(z)
            traj_target.append(encoded[start:end + 1].mean(0))
        else:
            ts = torch.tensor(all_times, dtype=torch.float32, device=dev)
            ts = _make_increasing(ts)
            zt = model._integrate(z, env, ts)

            for i in range(end - start + 1):
                t_i = times[start + i]
                best_idx = 0
                best_diff = abs(all_times[0] - t_i)
                for idx_t, at in enumerate(all_times):
                    diff = abs(at - t_i)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx_t
                if best_idx < len(zt):
                    traj_z.append(zt[best_idx])
                    traj_target.append(encoded[start + i])
            z = zt[-1]

    if t_target > bounds[-1] + 1e-6:
        last_start, last_end = segs[-1]
        env = encoded[last_start:last_end + 1].mean(0)
        if use_gru and gru_module is not None:
            dt = t_target - bounds[-1]
            steps = max(1, int(dt * 100))
            zt = gru_module(z, env, steps)
            z = zt[-1]
        else:
            ts = _make_increasing(
                torch.tensor([bounds[-1], t_target], dtype=torch.float32, device=dev))
            zt = model._integrate(z, env, ts)
            z = zt[-1]

    return z, traj_z, traj_target


class APTODE_NoAP(APTODE):
    """w/o AP: replace adaptive partitioning with fixed windows."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._fixed_part = FixedWindowPartitioner(self.w)

    def _evolve_single(self, uid, items, times, t_target):
        return _evolve_with_partitioner(
            self, self._fixed_part.partition, uid, items, times, t_target,
            reset_state_each_seg=False, use_gru=False)


class APTODE_NoODE(APTODE):
    """w/o ODE: replace ODE solver with GRU-based discrete update."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._gru = GRUEncoder(self.d, 128)

    def _evolve_single(self, uid, items, times, t_target):
        def _apt_part(embs, times):
            return apt_partition(embs, times, self.w, self.delta)
        return _evolve_with_partitioner(
            self, _apt_part, uid, items, times, t_target,
            reset_state_each_seg=False, use_gru=True, gru_module=self._gru)


class APTODE_NoSI(APTODE):
    """w/o SI: re-initialize with first item embedding at each segment."""
    def _evolve_single(self, uid, items, times, t_target):
        def _apt_part(embs, times):
            return apt_partition(embs, times, self.w, self.delta)
        return _evolve_with_partitioner(
            self, _apt_part, uid, items, times, t_target,
            reset_state_each_seg=True, use_gru=False)


# ======================================================================
# Part B: Divergence functions for design-choice ablation
# ======================================================================

def skl_divergence(p, q):
    """Symmetric KL divergence."""
    p = (p + 1e-8); p = p / p.sum(-1, keepdim=True)
    q = (q + 1e-8); q = q / q.sum(-1, keepdim=True)
    return (0.5 * (p * (p / q).log()).sum(-1) + 0.5 * (q * (q / p).log()).sum(-1)).item()


def wasserstein_1d(p, q):
    """1D Wasserstein distance approximation via sorting."""
    p_sorted, _ = torch.sort(p)
    q_sorted, _ = torch.sort(q)
    return torch.abs(p_sorted - q_sorted).mean().item()


# ======================================================================
# Part C: Readout functions for design-choice ablation
# ======================================================================

class GRUReadout(nn.Module):
    """GRU-based environment readout function."""
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)

    def forward(self, encoded):
        out, _ = self.gru(encoded.unsqueeze(0))
        return out.squeeze(0).mean(0)


class TransformerReadout(nn.Module):
    """Transformer-based environment readout function."""
    def __init__(self, d, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)

    def forward(self, encoded):
        out, _ = self.attn(encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0))
        return out.squeeze(0).mean(0)


class APTODE_GRUReadout(APTODE):
    """Replace mean pooling with GRU readout."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._gru_readout = GRUReadout(self.d)

    def _evolve_single(self, uid, items, times, t_target):
        if items.dim() == 0:
            items = items.unsqueeze(0)
        dev = items.device
        encoded = self.enc(items)
        bounds, segs = apt_partition(encoded.detach(), times, self.w, self.delta)

        z = self.user_emb(uid)
        traj_z, traj_target = [], []

        for seg_idx, (start, end) in enumerate(segs):
            if start > end:
                continue
            env = self._gru_readout(encoded[start:end + 1])

            tau_start = bounds[seg_idx]
            tau_end = bounds[seg_idx + 1] if seg_idx + 1 < len(bounds) else bounds[-1]
            seg_interaction_times = times[start:end + 1]
            all_times = set([tau_start, tau_end])
            for t in seg_interaction_times:
                if tau_start <= t <= tau_end:
                    all_times.add(t)
            all_times = sorted(all_times)

            if len(all_times) < 2:
                traj_z.append(z)
                traj_target.append(encoded[start])
                continue

            ts = torch.tensor(all_times, dtype=torch.float32, device=dev)
            ts = _make_increasing(ts)
            zt = self._integrate(z, env, ts)

            for i in range(end - start + 1):
                t_i = times[start + i]
                best_idx = min(range(len(all_times)),
                               key=lambda j: abs(all_times[j] - t_i))
                if best_idx < len(zt):
                    traj_z.append(zt[best_idx])
                    traj_target.append(encoded[start + i])
            z = zt[-1]

        if t_target > bounds[-1] + 1e-6:
            last_start, last_end = segs[-1]
            env = self._gru_readout(encoded[last_start:last_end + 1])
            ts = _make_increasing(
                torch.tensor([bounds[-1], t_target], dtype=torch.float32, device=dev))
            zt = self._integrate(z, env, ts)
            z = zt[-1]

        return z, traj_z, traj_target


class APTODE_TransformerReadout(APTODE):
    """Replace mean pooling with Transformer readout."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tf_readout = TransformerReadout(self.d)

    def _evolve_single(self, uid, items, times, t_target):
        if items.dim() == 0:
            items = items.unsqueeze(0)
        dev = items.device
        encoded = self.enc(items)
        bounds, segs = apt_partition(encoded.detach(), times, self.w, self.delta)

        z = self.user_emb(uid)
        traj_z, traj_target = [], []

        for seg_idx, (start, end) in enumerate(segs):
            if start > end:
                continue
            env = self._tf_readout(encoded[start:end + 1])

            tau_start = bounds[seg_idx]
            tau_end = bounds[seg_idx + 1] if seg_idx + 1 < len(bounds) else bounds[-1]
            seg_interaction_times = times[start:end + 1]
            all_times = set([tau_start, tau_end])
            for t in seg_interaction_times:
                if tau_start <= t <= tau_end:
                    all_times.add(t)
            all_times = sorted(all_times)

            if len(all_times) < 2:
                traj_z.append(z)
                traj_target.append(encoded[start])
                continue

            ts = torch.tensor(all_times, dtype=torch.float32, device=dev)
            ts = _make_increasing(ts)
            zt = self._integrate(z, env, ts)

            for i in range(end - start + 1):
                t_i = times[start + i]
                best_idx = min(range(len(all_times)),
                               key=lambda j: abs(all_times[j] - t_i))
                if best_idx < len(zt):
                    traj_z.append(zt[best_idx])
                    traj_target.append(encoded[start + i])
            z = zt[-1]

        if t_target > bounds[-1] + 1e-6:
            last_start, last_end = segs[-1]
            env = self._tf_readout(encoded[last_start:last_end + 1])
            ts = _make_increasing(
                torch.tensor([bounds[-1], t_target], dtype=torch.float32, device=dev))
            zt = self._integrate(z, env, ts)
            z = zt[-1]

        return z, traj_z, traj_target


# ======================================================================
# L2 trajectory loss (replaces cosine)
# ======================================================================

def l2_dyn_loss(all_z, all_t, device):
    """L2 distance trajectory alignment loss (replaces cosine-based dyn_loss)."""
    parts = []
    for zs, ts in zip(all_z, all_t):
        if not zs or not ts:
            continue
        n = min(len(zs), len(ts))
        z_stack = torch.stack(zs[:n])
        t_stack = torch.stack(ts[:n])
        parts.append(F.mse_loss(z_stack, t_stack))
    if parts:
        return torch.stack(parts).mean()
    return torch.tensor(0., device=device)


# ======================================================================
# Training utilities (shared by main and design ablation)
# ======================================================================

def train_variant(name, model, ds, args, dev, use_l2_loss=False):
    """Train a single ablation variant for one seed."""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(PairwiseDataset(ds.train, ds.n_items),
                        batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)

    best_ndcg, stale = -1., 0
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0., 0
        t0 = time.time()

        for batch in tqdm(loader, desc=f'{name} ep{ep}', leave=False):
            batch = {k: v.to(dev) for k, v in batch.items()}
            try:
                sp, sn, tz, tt = model(batch)
                l_rec = bpr_loss(sp, sn)
                l_dyn = l2_dyn_loss(tz, tt, dev) if use_l2_loss else dyn_loss(tz, tt, dev)
                loss = l_rec + args.alpha * l_dyn

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                ep_loss += loss.item()
                nb += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.warning('OOM, skipped batch')
                else:
                    log.warning(f'batch error: {e}')

        dt = time.time() - t0
        if nb > 0:
            log.info(f'  {name} ep {ep}/{args.epochs} loss={ep_loss/nb:.4f} {dt:.0f}s')
        else:
            log.warning(f'  {name} ep {ep}: all batches failed')

        # Early stopping on validation NDCG@10
        if ep % 3 == 0:
            n_eval = min(args.eval_users, len(ds.val))
            m = run_eval(model, ds.val, ds.train, ds.n_items, max_u=n_eval)
            if m['N@10'] > best_ndcg:
                best_ndcg = m['N@10']; stale = 0
            else:
                stale += 3
            if stale >= args.patience:
                log.info(f'  {name} early stop at ep {ep}')
                break

    return run_eval(model, ds.test, ds.train, ds.n_items,
                    max_u=min(args.eval_users, len(ds.test)))


# ======================================================================
# Main entry points
# ======================================================================

def run_main_ablation(args):
    """Run main component ablation with multi-seed support."""
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')
    ds = RecDataset(args.dataset, args.data_dir, args.core)

    def _make_model(cls):
        return cls(ds.n_users, ds.n_items, args.d, args.h,
                   args.w, args.delta, args.atol, args.rtol).to(dev)

    variants = [
        ('APT-ODE (Full)', APTODE),
        ('w/o AP', APTODE_NoAP),
        ('w/o ODE', APTODE_NoODE),
        ('w/o SI', APTODE_NoSI),
    ]

    results = {}
    for name, cls in variants:
        log.info(f'\n=== {name} ===')
        metrics_list = []
        for si in range(args.n_seeds):
            seed = args.seed + si * 7
            set_seed(seed)
            model = _make_model(cls)
            m = train_variant(name, model, ds, args, dev)
            metrics_list.append(m)
            log.info(f'  seed {seed}: N@20={m["N@20"]:.4f} R@20={m["R@20"]:.4f}')

        if args.n_seeds > 1:
            agg = {}
            for key in ['R@10', 'N@10', 'R@20', 'N@20']:
                vals = [x[key] for x in metrics_list]
                agg[key] = np.mean(vals)
                agg[key + '_std'] = np.std(vals)
            agg['n'] = metrics_list[0]['n']
            results[name] = agg
        else:
            results[name] = metrics_list[0]
        log.info(f'{name}: N@20={results[name].get("N@20",0):.4f} '
                 f'R@20={results[name].get("R@20",0):.4f}')

    log.info('\n--- Main Ablation Summary (NDCG@20) ---')
    full_n20 = results.get('APT-ODE (Full)', {}).get('N@20', 0)
    for name, m in results.items():
        n20 = m.get('N@20', 0)
        if name != 'APT-ODE (Full)' and full_n20 > 0:
            drop = (full_n20 - n20) / full_n20 * 100
            log.info(f'  {name}: N@20={n20:.4f} (drop {drop:.2f}%)')
        else:
            log.info(f'  {name}: N@20={n20:.4f}')

    return results


def run_design_ablation(args):
    """Run design-choice ablation with multi-seed support."""
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')
    ds = RecDataset(args.dataset, args.data_dir, args.core)

    def _make_model(cls=APTODE, **extra_kw):
        return cls(ds.n_users, ds.n_items, args.d, args.h,
                   args.w, args.delta, args.atol, args.rtol, **extra_kw).to(dev)

    # (name, model, use_l2_loss)
    variants = [
        ('JSD (ours)',          _make_model(),                    False),
        ('JSD->SKL',            _make_model(divergence_fn=skl_divergence), False),
        ('JSD->WD',             _make_model(divergence_fn=wasserstein_1d), False),
        ('Mean->GRU',           _make_model(cls=APTODE_GRUReadout),        False),
        ('Mean->Transformer',   _make_model(cls=APTODE_TransformerReadout),False),
        ('Cosine->L2',          _make_model(),                    True),
    ]

    results = {}
    for name, model, use_l2 in variants:
        log.info(f'\n=== {name} ===')
        metrics_list = []
        for si in range(args.n_seeds):
            seed = args.seed + si * 7
            set_seed(seed)
            # Rebuild model for each seed to ensure independent initialization
            if 'SKL' in name:
                m = _make_model(divergence_fn=skl_divergence)
            elif 'WD' in name:
                m = _make_model(divergence_fn=wasserstein_1d)
            elif 'GRU' in name:
                m = _make_model(cls=APTODE_GRUReadout)
            elif 'Transformer' in name:
                m = _make_model(cls=APTODE_TransformerReadout)
            else:
                m = _make_model()
            met = train_variant(name, m, ds, args, dev, use_l2_loss=use_l2)
            metrics_list.append(met)
            log.info(f'  seed {seed}: N@20={met["N@20"]:.4f} R@20={met["R@20"]:.4f}')

        if args.n_seeds > 1:
            agg = {}
            for key in ['R@10', 'N@10', 'R@20', 'N@20']:
                vals = [x[key] for x in metrics_list]
                agg[key] = np.mean(vals)
                agg[key + '_std'] = np.std(vals)
            agg['n'] = metrics_list[0]['n']
            results[name] = agg
        else:
            results[name] = metrics_list[0]
        log.info(f'{name}: N@20={results[name].get("N@20",0):.4f} '
                 f'R@20={results[name].get("R@20",0):.4f}')

    log.info('\n--- Design-Choice Ablation Summary (NDCG@20) ---')
    base_n20 = results.get('JSD (ours)', {}).get('N@20', 0)
    for name, m in results.items():
        n20 = m.get('N@20', 0)
        if name != 'JSD (ours)' and base_n20 > 0:
            drop = (base_n20 - n20) / base_n20 * 100
            log.info(f'  {name}: N@20={n20:.4f} (drop {drop:.2f}%)')
        else:
            log.info(f'  {name}: N@20={n20:.4f}')

    return results


# ======================================================================
# CLI
# ======================================================================

def cli():
    p = argparse.ArgumentParser(description='APT-ODE ablation study')
    p.add_argument('--dataset', default='amazon')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--ablation', default='main',
                   choices=['main', 'design', 'all'],
                   help='main=component ablation, design=design-choice, all=both')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=5)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--patience', type=int, default=10, help='early stop patience')
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_seeds', type=int, default=1, help='number of random seeds')
    p.add_argument('--eval_users', type=int, default=500)

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    args = cli()
    if args.ablation in ('main', 'all'):
        run_main_ablation(args)
    if args.ablation in ('design', 'all'):
        run_design_ablation(args)
