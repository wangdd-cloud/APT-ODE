"""
Ablation study for APT-ODE (Section 5.4).
Compares: full model, w/o AP (fixed window), w/o SI (no state inheritance).
"""

import sys, logging, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from apt_ode import (APTODE, RecDataset, PairwiseDataset, pad_collate,
                     bpr_loss, dyn_loss, run_eval, set_seed,
                     apt_partition, _make_increasing)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class FixedWindowPartitioner:
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


def _evolve_with_partitioner(model, partitioner, uid, items, times, t_target,
                              reset_state_each_seg=False):
    """
    Shared evolution logic for ablation variants.
    Uses boundary-to-boundary integration matching Algorithm 2.
    
    partitioner: callable with .partition(embs, times) -> bounds, segs
    reset_state_each_seg: if True, reinit z at each segment (w/o SI)
    """
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

        # w/o SI: random re-init at each new segment
        if reset_state_each_seg and seg_idx > 0:
            z = torch.randn(model.d, device=dev) * 0.1

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
        ts = _make_increasing(
            torch.tensor([bounds[-1], t_target], dtype=torch.float32, device=dev))
        zt = model._integrate(z, env, ts)
        z = zt[-1]

    return z, traj_z, traj_target


class APTODE_NoAP(APTODE):
    """Replace adaptive partitioning with fixed windows."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._fixed_part = FixedWindowPartitioner(self.w)

    def _evolve_single(self, uid, items, times, t_target):
        return _evolve_with_partitioner(
            self, self._fixed_part.partition, uid, items, times, t_target,
            reset_state_each_seg=False)


class APTODE_NoSI(APTODE):
    """Disable state inheritance: random re-init at each segment boundary."""
    def _evolve_single(self, uid, items, times, t_target):
        def _apt_part(embs, times):
            return apt_partition(embs, times, self.w, self.eta)
        return _evolve_with_partitioner(
            self, _apt_part, uid, items, times, t_target,
            reset_state_each_seg=True)


def train_variant(name, model, ds, args, dev):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(PairwiseDataset(ds.train, ds.n_items),
                        batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0., 0
        t0 = time.time()

        for batch in tqdm(loader, desc=f'{name} ep{ep}', leave=False):
            batch = {k: v.to(dev) for k, v in batch.items()}
            try:
                sp, sn, tz, tt = model(batch)
                l_rec = bpr_loss(sp, sn)
                l_dyn = dyn_loss(tz, tt, dev)
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

    return run_eval(model, ds.test, ds.train, ds.n_items,
                    max_u=min(args.eval_users, len(ds.test)))


def run_ablation(args):
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    ds = RecDataset(args.dataset, args.data_dir, args.core)

    variants = [
        ('APT-ODE (Full)', APTODE),
        ('w/o AP', APTODE_NoAP),
        ('w/o SI', APTODE_NoSI),
    ]

    results = {}
    for name, cls in variants:
        log.info(f'\n=== {name} ===')
        model = cls(ds.n_users, ds.n_items, args.d, args.h,
                    args.w, args.eta, args.atol, args.rtol).to(dev)
        m = train_variant(name, model, ds, args, dev)
        results[name] = m
        log.info(f'{name}: R@10={m["R@10"]:.4f} N@10={m["N@10"]:.4f} '
                 f'R@20={m["R@20"]:.4f} N@20={m["N@20"]:.4f}')

    log.info('\n--- Ablation Summary ---')
    full_n20 = results.get('APT-ODE (Full)', {}).get('N@20', 0)
    for name, m in results.items():
        n20 = m.get('N@20', 0)
        if name != 'APT-ODE (Full)' and full_n20 > 0:
            drop = (full_n20 - n20) / full_n20 * 100
            log.info(f'  {name}: N@20={n20:.4f} (drop {drop:.2f}%)')
        else:
            log.info(f'  {name}: N@20={n20:.4f}')


def cli():
    p = argparse.ArgumentParser(description='APT-ODE ablation study')
    p.add_argument('--dataset', default='synthetic')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=5)
    p.add_argument('--eta', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_users', type=int, default=300)

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    run_ablation(cli())
