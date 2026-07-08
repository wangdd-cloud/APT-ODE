"""
Interpretability analysis for APT-ODE (RQ4 / Section 4.4).

Computes all statistics reported in the paper:
  - Average segments per user (3.8 / 4.2 / 5.1)
  - Segment distribution (1 seg / 2-5 segs / 6+ segs)
  - Average segment length (2.3 / 3.0 / 28.3)
  - Quartile analysis (2.1x more segments in highest vs lowest activity quartile)
  - Large-gap boundary proportion (76% / 71% / 68%)
  - Non-gap boundary category-change proportion (78% / 71% / 65%)
  - Short-sequence user proportion (51.7% / 37.3% / 2.1%)
  - Short-sequence NDCG@20 (0.0812 on Amazon)
  - Average product categories per user (3.4 on Amazon)

Data requirements (all files placed under --data_dir):
  Mandatory (interaction data):
    Amazon: Electronics_5.json.gz
    Steam:  steam_reviews.json.gz
    ML-20M: ratings.csv

  Optional (item metadata; skipped with a warning if absent):
    Amazon: meta_Electronics.json.gz   → top-level product category
    Steam:  steam_games.json.gz        → primary genre
    ML-20M: movies.csv                 → primary genre

  The script loads interaction data twice:
    1. Via RecDataset (apt_ode.py)         → model-compatible integer item IDs
    2. Via load_raw_data (this file)        → string item IDs for category lookup
  The second pass reconstructs the same string→int mapping as RecDataset,
  then inverts it (int→string) to cross-reference with category metadata.
  Both passes must use the same version of the raw data and the same k-core
  filtering (--core) to keep the two mappings consistent.

  A trained model checkpoint (--model_path) is required for meaningful segment
  statistics and short-sequence evaluation.  Without one, the script falls back
  to a randomly initialized encoder whose boundaries will be random.

Usage:
  python analysis.py --dataset amazon --data_dir ./data/ \\
                     --model_path aptode_amazon.pt
"""

import sys, os, json, gzip, logging, argparse
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from apt_ode import (RecDataset, APTODE, apt_partition, _median_gap,
                     set_seed, _safe_load)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading with preserved string→int mappings
# ---------------------------------------------------------------------------

def load_raw_data(dataset_name, data_dir, core):
    """Load raw interaction data with string IDs preserved.
    Returns (raw_data, item_set) where raw_data is {user_str: [(item_str, timestamp), ...]}.
    """
    from apt_ode import _kcore
    data = defaultdict(list)

    if dataset_name == 'amazon':
        fp = os.path.join(data_dir, 'Electronics_5.json.gz')
        if not os.path.exists(fp):
            raise FileNotFoundError(f'Dataset file not found: {fp}')
        log.info(f'loading {fp}')
        with gzip.open(fp, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    data[r['reviewerID']].append((r['asin'], float(r['unixReviewTime'])))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    elif dataset_name == 'steam':
        fp = os.path.join(data_dir, 'steam_reviews.json.gz')
        if not os.path.exists(fp):
            raise FileNotFoundError(f'Dataset file not found: {fp}')
        log.info(f'loading {fp}')
        with gzip.open(fp, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    u = r.get('username', '')
                    i = str(r.get('product_id', ''))
                    t = float(r.get('date', 0))
                    if u and i:
                        data[u].append((i, t))
                except (json.JSONDecodeError, ValueError):
                    pass

    elif dataset_name == 'ml20m':
        fp = os.path.join(data_dir, 'ratings.csv')
        if not os.path.exists(fp):
            raise FileNotFoundError(f'Dataset file not found: {fp}')
        log.info(f'loading {fp}')
        with open(fp) as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4:
                    data[p[0]].append((p[1], float(p[3])))
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # Sort by timestamp
    for u in data:
        data[u].sort(key=lambda x: x[1])

    # k-core filtering
    if core > 0:
        from apt_ode import _kcore
        data = _kcore(dict(data), core)

    return dict(data)


def load_item_categories(dataset_name, data_dir):
    """Load item → category label mapping.

    Args:
        dataset_name: 'amazon' | 'steam' | 'ml20m'
        data_dir: path to data directory

    Returns:
        dict: item_string_id → category_name, or empty dict if unavailable.
    """
    cat_map = {}

    if dataset_name == 'amazon':
        meta_path = os.path.join(data_dir, 'meta_Electronics.json.gz')
        if not os.path.exists(meta_path):
            log.warning(f'metadata not found: {meta_path} — category analysis skipped')
            return {}
        log.info(f'loading categories from {meta_path}')
        with gzip.open(meta_path, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    asin = r.get('asin', '')
                    cats = r.get('categories', [])
                    if cats and len(cats) > 0:
                        # First entry of the first sublist = top-level category
                        first = cats[0]
                        if isinstance(first, list) and len(first) > 0:
                            top_cat = str(first[0])
                        else:
                            top_cat = str(first)
                        cat_map[asin] = top_cat
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info(f'  {len(cat_map)} items with category labels')

    elif dataset_name == 'steam':
        meta_path = os.path.join(data_dir, 'steam_games.json.gz')
        if not os.path.exists(meta_path):
            log.warning(f'metadata not found: {meta_path} — category analysis skipped')
            return {}
        log.info(f'loading genres from {meta_path}')
        with gzip.open(meta_path, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    pid = str(r.get('id', ''))
                    genres = r.get('genres', [])
                    cat_map[pid] = str(genres[0]) if genres else 'Unknown'
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info(f'  {len(cat_map)} items with genre labels')

    elif dataset_name == 'ml20m':
        meta_path = os.path.join(data_dir, 'movies.csv')
        if not os.path.exists(meta_path):
            log.warning(f'metadata not found: {meta_path} — category analysis skipped')
            return {}
        log.info(f'loading genres from {meta_path}')
        with open(meta_path, encoding='utf-8') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    movie_id = parts[0]
                    genres = parts[-1].split('|')
                    cat_map[movie_id] = genres[0] if genres else 'Unknown'
        log.info(f'  {len(cat_map)} movies with genre labels')

    return cat_map


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_full_analysis(args):
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    # ---- 1. Load dataset (RecDataset for model-compatible indexing) ----
    ds = RecDataset(args.dataset, args.data_dir, args.core)
    log.info(f'Dataset: users={ds.n_users}, items={ds.n_items}')

    # ---- 2. Load model ----
    model = APTODE(ds.n_users, ds.n_items, args.d, args.h,
                   args.w, args.delta, args.atol, args.rtol).to(dev)

    trained = False
    if args.model_path and os.path.exists(args.model_path):
        ckpt = _safe_load(args.model_path, dev)
        model.load_state_dict(ckpt, strict=False)
        log.info(f'loaded model from {args.model_path}')
        trained = True
    else:
        log.warning('no checkpoint — using randomly initialized encoder '
                    '(segment stats will be meaningless)')

    # ---- 3. Build item reverse-mapping (int index → string ID) ----
    # Re-load raw data and construct the same mapping as RecDataset
    raw_data = load_raw_data(args.dataset, args.data_dir, args.core)

    # Build u2i and i2i exactly as RecDataset does
    u2i, i2i = {}, {}
    uid, iid = 0, 1
    for u, s in raw_data.items():
        if u not in u2i:
            u2i[u] = uid; uid += 1
        for item, ts in s:
            if item not in i2i:
                i2i[item] = iid; iid += 1

    # Reverse: int → string
    idx2str = {v: k for k, v in i2i.items()}

    log.info(f'item mapping: {len(i2i)} string IDs → {iid} int indices, '
             f'{len(idx2str)} reverse entries')

    # ---- 4. Load category metadata ----
    cat_map = load_item_categories(args.dataset, args.data_dir)

    # ---- 5. Analyze every user in the training set ----
    model.eval()
    w = args.w

    # Per-user accumulators
    seg_counts = []          # K per user
    seg_lengths_all = []     # interactions per segment
    n_interactions = []      # n per user

    # Per-boundary accumulators
    total_boundaries = 0
    gap_boundaries = 0
    nongap_boundaries = 0
    nongap_cat_changes = 0

    # Category diversity: unique categories per user
    user_cat_counts = []

    # Short-sequence users for evaluation
    short_users = []

    for uid_int, seq in tqdm(ds.train.items(), desc='analyzing users'):

        items_int = [s[0] for s in seq]
        times = [s[1] for s in seq]
        n = len(items_int)
        n_interactions.append(n)

        # Map internal item IDs to string IDs for category lookup
        items_str = [idx2str.get(ii, '') for ii in items_int]

        # Count unique categories for this user
        user_cats = set()
        for istr in items_str:
            if istr in cat_map:
                user_cats.add(cat_map[istr])
        if user_cats:
            user_cat_counts.append(len(user_cats))

        # Short-sequence check
        if n < 2 * w:
            seg_counts.append(1)
            seg_lengths_all.append(n)
            short_users.append(uid_int)
            continue

        # Compute inter-interaction gaps and 75th percentile
        gaps = [times[i] - times[i - 1] for i in range(1, n)]
        p75 = np.percentile(gaps, 75) if gaps else float('inf')

        # Encode items and run APT partition
        items_t = torch.LongTensor(items_int).to(dev)
        with torch.no_grad():
            encoded = model.enc(items_t)
        bounds, segs = apt_partition(encoded.detach(), times, w, model.delta,
                                     model.divergence_fn)

        K = len(segs)
        seg_counts.append(K)

        for sidx, (start, end) in enumerate(segs):
            seg_lengths_all.append(end - start + 1)

        # Per-boundary analysis
        for k_idx in range(1, len(bounds) - 1):
            boundary_time = bounds[k_idx]
            total_boundaries += 1

            # Find items immediately left and right of the boundary
            left_idx = max(i for i in range(n) if times[i] <= boundary_time)
            right_candidates = [i for i in range(n) if times[i] > boundary_time]
            if not right_candidates:
                continue
            right_idx = right_candidates[0]

            # Determine if this is a large-gap boundary
            gap_val = times[right_idx] - times[left_idx] if right_idx > left_idx else 0
            is_large_gap = gap_val > p75

            if is_large_gap:
                gap_boundaries += 1
            else:
                nongap_boundaries += 1
                # Check category change for non-gap boundaries
                left_cat = cat_map.get(items_str[left_idx], None) if left_idx < len(items_str) else None
                right_cat = cat_map.get(items_str[right_idx], None) if right_idx < len(items_str) else None
                if left_cat is not None and right_cat is not None and left_cat != right_cat:
                    nongap_cat_changes += 1

    # ---- 6. Compute aggregate statistics ----
    stats = {}
    n_users = len(seg_counts)
    seg_arr = np.array(seg_counts)
    n_arr = np.array(n_interactions)

    stats['n_users'] = n_users
    stats['avg_segments'] = np.mean(seg_arr)
    stats['pct_1seg'] = 100 * np.mean(seg_arr == 1)
    stats['pct_2_5seg'] = 100 * np.mean((seg_arr >= 2) & (seg_arr <= 5))
    stats['pct_6plus_seg'] = 100 * np.mean(seg_arr >= 6)
    stats['avg_seg_len'] = np.mean(seg_lengths_all) if seg_lengths_all else 0

    # Quartile analysis
    if n_users >= 4:
        q25 = np.percentile(n_arr, 25)
        q75 = np.percentile(n_arr, 75)
        low_mask = n_arr <= q25
        high_mask = n_arr >= q75
        low_avg = np.mean(seg_arr[low_mask]) if low_mask.sum() > 0 else 0
        high_avg = np.mean(seg_arr[high_mask]) if high_mask.sum() > 0 else 0
        ratio = high_avg / low_avg if low_avg > 0 else 0
    else:
        low_avg, high_avg, ratio = 0, 0, 0

    stats['q_low_avg_segs'] = low_avg
    stats['q_high_avg_segs'] = high_avg
    stats['quartile_ratio'] = ratio

    # Boundary analysis
    stats['total_boundaries'] = total_boundaries
    if total_boundaries > 0:
        stats['pct_gap_boundaries'] = 100 * gap_boundaries / total_boundaries
    else:
        stats['pct_gap_boundaries'] = 0
    if nongap_boundaries > 0:
        stats['pct_nongap_cat_change'] = 100 * nongap_cat_changes / nongap_boundaries
        stats['nongap_total'] = nongap_boundaries
        stats['nongap_cat_changes'] = nongap_cat_changes
    else:
        stats['pct_nongap_cat_change'] = 0
        stats['nongap_total'] = 0
        stats['nongap_cat_changes'] = 0

    # Short-sequence users
    stats['n_short_seq'] = len(short_users)
    stats['pct_short_seq'] = 100 * len(short_users) / n_users if n_users > 0 else 0

    # Category diversity
    stats['avg_categories_per_user'] = (np.mean(user_cat_counts)
                                        if user_cat_counts else 0)
    stats['n_users_with_cats'] = len(user_cat_counts)

    # ---- 7. Short-sequence evaluation (if model is trained) ----
    if trained and short_users:
        log.info(f'\nEvaluating {len(short_users)} short-sequence users...')
        model.eval()
        model.reset_timing()

        # Build a filtered evaluation: only short-sequence users in test set
        hits_10, ndcg_10, ndcg_20, cnt = 0.0, 0.0, 0.0, 0

        for u in tqdm(short_users, desc='short-seq eval'):
            if u not in ds.test:
                continue
            hist, (gt, gt_t) = ds.test[u]
            if len(hist) < 2:
                continue
            hi = [h[0] for h in hist]
            ht = [h[1] for h in hist]
            t_next = ht[-1] + _median_gap(ht)
            mask_set = {s[0] for s in ds.train.get(u, [])} | {h[0] for h in hist}

            try:
                sc = model.score_all(u, hi, ht, t_next)
            except RuntimeError:
                continue

            for it in mask_set:
                if it < len(sc):
                    sc[it] = -np.inf
            sc[0] = -np.inf

            rank = np.argsort(-sc)
            for k in [10, 20]:
                topk = rank[:k]
                if gt in topk:
                    if k == 10:
                        hits_10 += 1.0
                    pos = np.where(topk == gt)[0][0]
                    ndcg_val = 1.0 / np.log2(pos + 2)
                    if k == 10:
                        ndcg_10 += ndcg_val
                    if k == 20:
                        ndcg_20 += ndcg_val
            cnt += 1

        if cnt > 0:
            stats['short_seq_R10'] = hits_10 / cnt
            stats['short_seq_N10'] = ndcg_10 / cnt
            stats['short_seq_N20'] = ndcg_20 / cnt
            stats['short_seq_eval_n'] = cnt
        else:
            stats['short_seq_R10'] = 0
            stats['short_seq_N10'] = 0
            stats['short_seq_N20'] = 0
            stats['short_seq_eval_n'] = 0

    # ---- 8. Print results ----
    log.info(f'\n{"="*65}')
    log.info(f'  Interpretability Analysis — {args.dataset}')
    log.info(f'{"="*65}')
    log.info(f'  Users analyzed:                              {n_users}')
    log.info(f'')
    log.info(f'  Average segments per user:                  {stats["avg_segments"]:.1f}')
    log.info(f'  Average segment length (interactions):       {stats["avg_seg_len"]:.1f}')
    log.info(f'')
    log.info(f'  Segment distribution:')
    log.info(f'    1 segment:                                {stats["pct_1seg"]:.1f}%')
    log.info(f'    2–5 segments:                             {stats["pct_2_5seg"]:.1f}%')
    log.info(f'    6+ segments:                              {stats["pct_6plus_seg"]:.0f}%')
    log.info(f'')
    log.info(f'  Quartile analysis:')
    log.info(f'    Low-activity quartile avg segments:        {stats["q_low_avg_segs"]:.2f}')
    log.info(f'    High-activity quartile avg segments:       {stats["q_high_avg_segs"]:.2f}')
    log.info(f'    Ratio (high / low):                        {stats["quartile_ratio"]:.1f}x')
    log.info(f'')
    log.info(f'  Boundary analysis:')
    log.info(f'    Total boundaries detected:                 {stats["total_boundaries"]}')
    log.info(f'    At large timestamp gaps:                   {stats["pct_gap_boundaries"]:.0f}%')
    if stats["nongap_total"] > 0:
        log.info(f'    Non-gap boundaries with category change:   {stats["pct_nongap_cat_change"]:.0f}%'
                 f'  ({stats["nongap_cat_changes"]}/{stats["nongap_total"]})')
    else:
        log.info(f'    Non-gap category change:                   N/A (no metadata)')
    log.info(f'')
    log.info(f'  Short-sequence users (<2w={2*w} interactions):')
    log.info(f'    Count:                                     {stats["n_short_seq"]}')
    log.info(f'    Proportion:                                {stats["pct_short_seq"]:.1f}%')
    if trained and 'short_seq_N20' in stats:
        log.info(f'    APT-ODE NDCG@20 (short-seq users):         {stats["short_seq_N20"]:.4f}')
        log.info(f'    Eval users:                                {stats["short_seq_eval_n"]}')
    log.info(f'')
    if stats['n_users_with_cats'] > 0:
        log.info(f'  Avg product categories per user:             {stats["avg_categories_per_user"]:.1f}')
    log.info(f'{"="*65}')

    return stats


def cli():
    p = argparse.ArgumentParser(description='APT-ODE interpretability analysis (RQ4)')
    p.add_argument('--dataset', default='amazon',
                   help='amazon / steam / ml20m')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--model_path', default='',
                   help='path to trained APT-ODE checkpoint (.pt)')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--h', type=int, default=128)
    p.add_argument('--w', type=int, default=5)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    run_full_analysis(cli())
