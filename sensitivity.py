"""
Hyperparameter sensitivity analysis for APT-ODE (Section 5.5).
Analyzes impact of JSD threshold eta and window size w.
"""

import sys, logging, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from apt_ode import (APTODE, RecDataset, PairwiseDataset, pad_collate,
                     bpr_loss, dyn_loss, run_eval, set_seed)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def quick_train_and_eval(label, ds, model_kwargs, args, dev):
    model = APTODE(**model_kwargs).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(PairwiseDataset(ds.train, ds.n_items),
                        batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0., 0
        for batch in tqdm(loader, desc=f'{label} ep{ep}', leave=False):
            batch = {k: v.to(dev) for k, v in batch.items()}
            try:
                sp, sn, tz, tt = model(batch)
                loss = bpr_loss(sp, sn) + args.alpha * dyn_loss(tz, tt, dev)
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

        if nb > 0 and (ep % 5 == 0 or ep == args.epochs):
            log.info(f'  {label} ep {ep}/{args.epochs} loss={ep_loss/nb:.4f}')

    if nb == 0:
        log.warning(f'{label}: all training batches failed')

    return run_eval(model, ds.test, ds.train, ds.n_items,
                    max_u=min(args.eval_users, len(ds.test)))


def run_sensitivity(args):
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    ds = RecDataset(args.dataset, args.data_dir, args.core)

    base_kwargs = dict(
        n_users=ds.n_users, n_items=ds.n_items,
        d=args.d, h=args.h,
        atol=args.atol, rtol=args.rtol,
    )

    # --- eta sensitivity (fixed w=5) ---
    log.info('\n--- eta sensitivity (fixed w=5) ---')
    eta_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    eta_results = {}
    for eta in eta_values:
        label = f'eta={eta:.1f}'
        kwargs = {**base_kwargs, 'w': 5, 'eta': eta}
        m = quick_train_and_eval(label, ds, kwargs, args, dev)
        eta_results[eta] = m
        log.info(f'  eta={eta:.1f} -> N@20={m["N@20"]:.4f} R@20={m["R@20"]:.4f}')

    log.info('eta summary:')
    for eta, m in eta_results.items():
        log.info(f'  eta={eta:.1f}: N@10={m["N@10"]:.4f} N@20={m["N@20"]:.4f}')

    # --- w sensitivity (fixed eta=0.5) ---
    log.info('\n--- w sensitivity (fixed eta=0.5) ---')
    w_values = [3, 4, 5, 6, 7]
    w_results = {}
    for w in w_values:
        label = f'w={w}'
        kwargs = {**base_kwargs, 'w': w, 'eta': 0.5}
        m = quick_train_and_eval(label, ds, kwargs, args, dev)
        w_results[w] = m
        log.info(f'  w={w} -> N@20={m["N@20"]:.4f} R@20={m["R@20"]:.4f}')

    log.info('w summary:')
    for w, m in w_results.items():
        log.info(f'  w={w}: N@10={m["N@10"]:.4f} N@20={m["N@20"]:.4f}')


def cli():
    p = argparse.ArgumentParser(description='APT-ODE sensitivity analysis')
    p.add_argument('--dataset', default='synthetic')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_users', type=int, default=300)

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    run_sensitivity(cli())
