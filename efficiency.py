"""
Efficiency measurement for APT-ODE (RQ5 / Table 6).
Reports wall-clock training time per epoch, inference latency per user,
and peak GPU memory usage on ML-20M.
"""
import sys, time, logging, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from apt_ode import (APTODE, RecDataset, PairwiseDataset, pad_collate,
                     bpr_loss, dyn_loss, run_eval, set_seed, _safe_load,
                     _median_gap)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def measure_gpu_memory(dev='cuda'):
    """Return current GPU memory usage in GB."""
    if dev == 'cuda' and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def measure_training_efficiency(model_class, ds, args, dev, label='Model'):
    """Measure training time per epoch, inference latency, GPU memory.

    Returns dict with:
        train_time_per_epoch: seconds
        inference_ms_per_user: milliseconds
        gpu_memory_gb: peak GPU memory
    """
    set_seed(args.seed)

    # Build model
    model = model_class(ds.n_users, ds.n_items, args.d, args.h,
                        args.w, args.delta, args.atol, args.rtol).to(dev)

    if args.pretrained_emb:
        emb = _safe_load(args.pretrained_emb, dev)
        model.load_pretrained_embeddings(emb)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f'{label}: params={n_params:,}')

    # --- Training time per epoch (warmup + measure) ---
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_ds = PairwiseDataset(ds.train, ds.n_items)
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)

    # Warmup: 1 epoch
    log.info(f'{label}: warmup epoch...')
    model.train()
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        try:
            sp, sn, tz, tt = model(batch)
            loss = bpr_loss(sp, sn) + args.alpha * dyn_loss(tz, tt, dev)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        except RuntimeError:
            continue

    # Measure: 5 epochs
    train_times = []
    for ep in range(5):
        if dev == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            try:
                sp, sn, tz, tt = model(batch)
                loss = bpr_loss(sp, sn) + args.alpha * dyn_loss(tz, tt, dev)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            except RuntimeError:
                continue
        if dev == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_times.append(dt)
        log.info(f'{label} ep {ep+1}: {dt:.1f}s')

    avg_train_time = np.mean(train_times)

    # --- GPU memory ---
    if dev == 'cuda':
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        gpu_mem = 0.0

    # --- Inference latency ---
    model.eval()
    n_eval = min(200, len(ds.test))
    users = list(ds.test.keys())[:n_eval]

    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    cnt = 0
    for u in users:
        hist, (gt, gt_t) = ds.test[u]
        if len(hist) < 2:
            continue
        hi = [h[0] for h in hist]
        ht = [h[1] for h in hist]
        t_next = ht[-1] + _median_gap(ht)
        try:
            model.score_all(u, hi, ht, t_next)
            cnt += 1
        except RuntimeError:
            continue
    if dev == 'cuda':
        torch.cuda.synchronize()
    inf_time = time.time() - t0
    inf_ms = (inf_time / cnt * 1000) if cnt > 0 else 0.0

    log.info(f'{label}: train={avg_train_time:.1f}s/epoch, '
             f'inference={inf_ms:.2f}ms/user, GPU={gpu_mem:.1f}GB')

    return {
        'model': label,
        'train_time_per_epoch': avg_train_time,
        'inference_ms_per_user': inf_ms,
        'gpu_memory_gb': gpu_mem,
        'n_params': n_params,
    }


def run_efficiency(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    # Load ML-20M
    ds = RecDataset(args.dataset, args.data_dir, args.core)
    log.info(f'Dataset: {args.dataset}, users={ds.n_users}, items={ds.n_items}')

    results = []

    # Measure APT-ODE
    r = measure_training_efficiency(APTODE, ds, args, dev, label='APT-ODE')
    results.append(r)

    # Print summary table
    log.info(f'\n{"="*70}')
    log.info(f'{"Model":<20} {"Train(s/epoch)":>15} {"Infer(ms/user)":>15} {"GPU(GB)":>10}')
    log.info(f'{"-"*70}')
    for r in results:
        log.info(f'{r["model"]:<20} {r["train_time_per_epoch"]:>15.1f} '
                 f'{r["inference_ms_per_user"]:>15.2f} {r["gpu_memory_gb"]:>10.1f}')

    return results


def cli():
    p = argparse.ArgumentParser(description='APT-ODE efficiency measurement')
    p.add_argument('--dataset', default='ml20m', help='dataset for efficiency test')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=5)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrained_emb', default='',
                   help='path to pretrained embedding .pt file')

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    run_efficiency(cli())
