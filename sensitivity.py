"""
Hyperparameter sensitivity analysis for APT-ODE (RQ3).
Analyzes impact of JSD threshold delta and window size w on all three datasets.
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
    """Train APT-ODE with given kwargs and return test metrics. Uses early stopping."""
    model = APTODE(**model_kwargs).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(PairwiseDataset(ds.train, ds.n_items),
                        batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)

    best_ndcg, stale, best_state = -1., 0, None

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

        if nb > 0 and (ep % 3 == 0 or ep == args.epochs):
            log.info(f'  {label} ep {ep}/{args.epochs} loss={ep_loss/nb:.4f}')

        # Early stopping on validation NDCG@10 (every 3 epochs)
        if ep % 3 == 0:
            n_eval = min(args.eval_users, len(ds.val))
            m = run_eval(model, ds.val, ds.train, ds.n_items, max_u=n_eval)
            if m['N@10'] > best_ndcg:
                best_ndcg = m['N@10']
                stale = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 3
            if stale >= args.patience:
                log.info(f'  {label} early stop at ep {ep}')
                break

    if nb == 0:
        log.warning(f'{label}: all training batches failed')

    if best_state:
        model.load_state_dict(best_state)
    return run_eval(model, ds.test, ds.train, ds.n_items,
                    max_u=min(args.eval_users, len(ds.test)))


def run_sensitivity(args):
    """Run sensitivity analysis on one or all three datasets."""
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    datasets = args.datasets.split(',') if args.datasets != 'all' \
               else ['amazon', 'steam', 'ml20m']

    all_results = {}

    for ds_name in datasets:
        log.info(f'\n{"="*60}')
        log.info(f'Dataset: {ds_name}')
        log.info(f'{"="*60}')

        ds = RecDataset(ds_name, args.data_dir, args.core)

        base_kwargs = dict(
            n_users=ds.n_users, n_items=ds.n_items,
            d=args.d, h=args.h,
            atol=args.atol, rtol=args.rtol,
        )

        # --- delta sensitivity (fixed w=5) ---
        log.info(f'\n--- {ds_name}: delta sensitivity (fixed w=5) ---')
        delta_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        delta_results = {}
        for delta in delta_values:
            label = f'{ds_name}_delta={delta:.1f}'
            kwargs = {**base_kwargs, 'w': 5, 'delta': delta}
            m = quick_train_and_eval(label, ds, kwargs, args, dev)
            delta_results[delta] = m
            log.info(f'  delta={delta:.1f} -> N@20={m["N@20"]:.4f} R@20={m["R@20"]:.4f}')

        log.info(f'{ds_name} delta summary (NDCG@20):')
        for delta, m in delta_results.items():
            log.info(f'  delta={delta:.1f}: N@10={m["N@10"]:.4f} N@20={m["N@20"]:.4f}')

        # --- w sensitivity (fixed delta=0.5) ---
        log.info(f'\n--- {ds_name}: w sensitivity (fixed delta=0.5) ---')
        w_values = [3, 4, 5, 6, 7]
        w_results = {}
        for w in w_values:
            label = f'{ds_name}_w={w}'
            kwargs = {**base_kwargs, 'w': w, 'delta': 0.5}
            m = quick_train_and_eval(label, ds, kwargs, args, dev)
            w_results[w] = m
            log.info(f'  w={w} -> N@20={m["N@20"]:.4f} R@20={m["R@20"]:.4f}')

        log.info(f'{ds_name} w summary (NDCG@20):')
        for w, m in w_results.items():
            log.info(f'  w={w}: N@10={m["N@10"]:.4f} N@20={m["N@20"]:.4f}')

        all_results[ds_name] = {'delta': delta_results, 'w': w_results}

    # Cross-dataset summary
    if len(datasets) > 1:
        log.info(f'\n{"="*60}')
        log.info('Cross-dataset optimal hyperparameters:')
        log.info(f'{"="*60}')
        for ds_name in datasets:
            dr = all_results[ds_name]['delta']
            wr = all_results[ds_name]['w']
            best_delta = max(dr, key=lambda d: dr[d]['N@20'])
            best_w = max(wr, key=lambda w: wr[w]['N@20'])
            log.info(f'  {ds_name}: best delta={best_delta} (N@20={dr[best_delta]["N@20"]:.4f}), '
                     f'best w={best_w} (N@20={wr[best_w]["N@20"]:.4f})')

    return all_results


def cli():
    p = argparse.ArgumentParser(description='APT-ODE sensitivity analysis')
    p.add_argument('--datasets', default='all',
                   help='comma-separated list: amazon,steam,ml20m, or "all"')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=8, help='early stop patience')
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
