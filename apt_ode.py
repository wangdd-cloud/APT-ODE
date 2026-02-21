"""
APT-ODE: Adaptive Partitioning in Time with Neural ODEs
for Modeling User Preference Shifts
"""

import os, sys, time, gzip, json, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from torchdiffeq import odeint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _kcore(interactions, k):
    for _ in range(200):
        before = sum(len(v) for v in interactions.values())
        interactions = {u: s for u, s in interactions.items() if len(s) >= k}
        ic = defaultdict(int)
        for s in interactions.values():
            for iid, _ in s:
                ic[iid] += 1
        keep = {i for i, c in ic.items() if c >= k}
        interactions = {u: [(i, t) for i, t in s if i in keep]
                        for u, s in interactions.items()}
        interactions = {u: s for u, s in interactions.items() if len(s) >= k}
        after = sum(len(v) for v in interactions.values())
        if before == after:
            break
    return interactions


def load_data(name, path, core):
    data = defaultdict(list)

    if name == 'amazon':
        fp = os.path.join(path, 'Electronics_5.json.gz')
        if not os.path.exists(fp):
            return None
        log.info(f'loading {fp}')
        with gzip.open(fp, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    data[r['reviewerID']].append((r['asin'], float(r['unixReviewTime'])))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    elif name == 'steam':
        fp = os.path.join(path, 'steam_reviews.json.gz')
        if not os.path.exists(fp):
            return None
        log.info(f'loading {fp}')
        with gzip.open(fp, 'rt') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    u = r.get('username', '')
                    i = r.get('product_id', '')
                    t = float(r.get('date', 0))
                    if u and i:
                        data[u].append((i, t))
                except (json.JSONDecodeError, ValueError):
                    pass

    elif name == 'ml20m':
        fp = os.path.join(path, 'ratings.csv')
        if not os.path.exists(fp):
            return None
        log.info(f'loading {fp}')
        with open(fp) as f:
            f.readline()
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 4:
                    data[p[0]].append((p[1], float(p[3])))
    else:
        return None

    for u in data:
        data[u].sort(key=lambda x: x[1])
    if core > 0:
        data = _kcore(dict(data), core)
    return dict(data)


def make_synthetic(n_users=2000, n_items=500, seed=42):
    rng = np.random.RandomState(seed)
    data = {}
    for u in range(n_users):
        n = max(7, rng.poisson(20))
        items = rng.randint(1, n_items, size=n)
        ts = np.sort(rng.exponential(86400., size=n).cumsum())
        data[str(u)] = [(int(items[i]), float(ts[i])) for i in range(n)]
    return data


class RecDataset:
    def __init__(self, name, data_dir, core):
        raw = load_data(name, data_dir, core)
        if raw is None:
            log.warning(f'{name} not found in {data_dir}, using synthetic')
            raw = make_synthetic()

        u2i, i2i = {}, {}
        uid, iid = 0, 1
        seqs = {}
        for u, s in raw.items():
            if u not in u2i:
                u2i[u] = uid; uid += 1
            ui = u2i[u]
            indexed = []
            for item, ts in s:
                if item not in i2i:
                    i2i[item] = iid; iid += 1
                indexed.append((i2i[item], ts))
            seqs[ui] = indexed

        self.n_users = uid
        self.n_items = iid

        for u in seqs:
            ts = [x[1] for x in seqs[u]]
            lo, hi = min(ts), max(ts)
            span = hi - lo if hi > lo else 1.
            seqs[u] = [(item, (t - lo) / span) for item, t in seqs[u]]

        self.train, self.val, self.test = {}, {}, {}
        for u, s in seqs.items():
            if len(s) < 3:
                continue
            self.train[u] = s[:-2]
            self.val[u] = (s[:-2], s[-2])
            self.test[u] = (s[:-1], s[-1])

        n_inter = sum(len(s) for s in seqs.values())
        log.info(f'dataset={name} users={self.n_users} items={self.n_items} '
                 f'interactions={n_inter} train_users={len(self.train)}')


class PairwiseDataset(Dataset):
    def __init__(self, train, n_items, maxlen=50):
        self.data = []
        for u, seq in train.items():
            positives = {s[0] for s in seq}
            for j in range(1, len(seq)):
                hist = seq[:j][-maxlen:]
                neg = random.randint(1, n_items - 1)
                while neg in positives:
                    neg = random.randint(1, n_items - 1)
                self.data.append((u, hist, seq[j][0], neg, seq[j][1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def pad_collate(batch):
    users, pos, neg, times = [], [], [], []
    items_l, times_l, lens = [], [], []
    for u, hist, p, n, t in batch:
        users.append(u); pos.append(p); neg.append(n); times.append(t)
        items_l.append([h[0] for h in hist])
        times_l.append([h[1] for h in hist])
        lens.append(len(hist))
    ml = max(lens)
    pi = [il + [0] * (ml - len(il)) for il in items_l]
    pt = [tl + [0.] * (ml - len(tl)) for tl in times_l]
    return {
        'u': torch.LongTensor(users), 'hi': torch.LongTensor(pi),
        'ht': torch.FloatTensor(pt), 'hl': torch.LongTensor(lens),
        'pos': torch.LongTensor(pos), 'neg': torch.LongTensor(neg),
        'pt': torch.FloatTensor(times),
    }


# ---------------------------------------------------------------------------
# model components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, n_items, d):
        super().__init__()
        self.emb = nn.Embedding(n_items, d, padding_idx=0)
        nn.init.xavier_uniform_(self.emb.weight[1:])
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, ids):
        return self.mlp(self.emb(ids))

    def raw(self, ids):
        return self.emb(ids)


class VectorField(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(2 * d, h), nn.Softplus(),
            nn.Linear(h, h), nn.Softplus(),
            nn.Linear(h, d),
        )

    def forward(self, t, ze):
        squeeze = ze.dim() == 1
        if squeeze:
            ze = ze.unsqueeze(0)
        z, e = ze[:, :self.d], ze[:, self.d:]
        dz = self.net(torch.cat([z, e], -1))
        out = torch.cat([dz, torch.zeros_like(e)], -1)
        return out.squeeze(0) if squeeze else out


def jsd(p, q):
    p = (p + 1e-8); p = p / p.sum(-1, keepdim=True)
    q = (q + 1e-8); q = q / q.sum(-1, keepdim=True)
    m = .5 * (p + q)
    return (.5 * (p * (p / m).log()).sum(-1) + .5 * (q * (q / m).log()).sum(-1)).item()


def apt_partition(embs, times, w, eta):
    n = len(times)
    if n < 2 * w:
        return [times[0], times[-1]], [(0, n - 1)]

    bounds = [times[0]]
    starts = [0]
    j = 0
    while j <= n - 2 * w:
        pl = F.softmax(embs[j:j + w].mean(0), dim=-1)
        pr = F.softmax(embs[j + w:j + 2 * w].mean(0), dim=-1)
        if jsd(pl.unsqueeze(0), pr.unsqueeze(0)) > eta:
            bounds.append((times[j + w - 1] + times[j + w]) / 2.)
            starts.append(j + w)
            j += w
        else:
            j += 1

    if times[-1] > bounds[-1]:
        bounds.append(times[-1])

    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else n - 1
        segs.append((s, e))
    return bounds, segs


def _make_increasing(t, gap=1e-6):
    t = t.clone()
    for i in range(1, len(t)):
        if t[i] - t[i - 1] < gap:
            t[i] = t[i - 1] + gap
    return t


def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class APTODE(nn.Module):
    def __init__(self, n_users, n_items, d=64, h=128, w=5, eta=0.5,
                 atol=1e-5, rtol=1e-5):
        super().__init__()
        self.d, self.w, self.eta = d, w, eta
        self.atol, self.rtol = atol, rtol
        self.n_items = n_items

        self.enc = Encoder(n_items, d)
        self.user_emb = nn.Embedding(n_users, d)
        nn.init.xavier_uniform_(self.user_emb.weight)
        self.vf = VectorField(d, h)

    def _integrate(self, z, e, tspan):
        aug = torch.cat([z, e])
        try:
            traj = odeint(self.vf, aug, tspan, method='dopri5',
                          atol=self.atol, rtol=self.rtol)
        except RuntimeError:
            try:
                traj = odeint(self.vf, aug, tspan, method='euler',
                              options={'step_size': 0.05})
            except RuntimeError:
                log.warning('ODE integration failed, returning initial state')
                traj = aug.unsqueeze(0).expand(len(tspan), -1)
        return traj[:, :self.d]

    def _evolve_single(self, uid, items, times, t_target):
        """
        Piecewise ODE evolution with state inheritance (Algorithm 2).
        Integrates from boundary to boundary using environment embeddings,
        ensuring continuous evolution across the full time axis.
        """
        if items.dim() == 0:
            items = items.unsqueeze(0)

        dev = items.device
        encoded = self.enc(items)
        bounds, segs = apt_partition(encoded.detach(), times, self.w, self.eta)

        z = self.user_emb(uid)
        traj_z, traj_target = [], []

        # Algorithm 2: integrate from tau_{k-1} to tau_k for each segment
        for seg_idx, (start, end) in enumerate(segs):
            if start > end:
                continue

            env = encoded[start:end + 1].mean(0)

            # boundary times for this segment: tau_{k-1} and tau_k
            tau_start = bounds[seg_idx]
            tau_end = bounds[seg_idx + 1] if seg_idx + 1 < len(bounds) else bounds[-1]

            # collect all time points within this segment:
            # boundary start, interaction times, boundary end
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

            zt = self._integrate(z, env, ts)

            # collect trajectory states at interaction time points for L_dyn
            for i in range(end - start + 1):
                t_i = times[start + i]
                # find closest index in all_times
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

            # state inheritance: terminal state at tau_k becomes initial for next
            z = zt[-1]

        # extrapolate to target time (Eq.9)
        if t_target > bounds[-1] + 1e-6:
            last_start, last_end = segs[-1]
            env = encoded[last_start:last_end + 1].mean(0)
            ts = _make_increasing(
                torch.tensor([bounds[-1], t_target], dtype=torch.float32, device=dev))
            zt = self._integrate(z, env, ts)
            z = zt[-1]

        return z, traj_z, traj_target

    def forward(self, batch):
        bsz = batch['u'].shape[0]
        dev = batch['u'].device
        z_out, all_tz, all_tt = [], [], []

        for b in range(bsz):
            L = batch['hl'][b].item()
            if L == 0:
                z_out.append(self.user_emb(batch['u'][b]))
                all_tz.append([]); all_tt.append([])
                continue

            z, tz, tt = self._evolve_single(
                batch['u'][b],
                batch['hi'][b, :L],
                batch['ht'][b, :L].tolist(),
                batch['pt'][b].item())
            z_out.append(z)
            all_tz.append(tz); all_tt.append(tt)

        z_final = torch.stack(z_out)
        v_pos = self.enc.raw(batch['pos'])
        v_neg = self.enc.raw(batch['neg'])
        s_pos = (z_final * v_pos).sum(-1)
        s_neg = (z_final * v_neg).sum(-1)
        return s_pos, s_neg, all_tz, all_tt

    @torch.no_grad()
    def score_all(self, uid, items, times, t_target):
        dev = next(self.parameters()).device

        uid_t = torch.LongTensor([uid]).to(dev).squeeze(0)
        items_t = torch.LongTensor(items).to(dev)
        if items_t.dim() == 0:
            items_t = items_t.unsqueeze(0)

        if isinstance(times, (list, tuple)):
            time_list = [float(x) for x in times]
        else:
            time_list = [float(times)]

        z, _, _ = self._evolve_single(uid_t, items_t, time_list, float(t_target))

        scores = []
        for s in range(0, self.n_items, 1024):
            e = min(s + 1024, self.n_items)
            ids = torch.arange(s, e, device=dev)
            emb = self.enc.raw(ids)
            scores.append((emb @ z).cpu())
        return torch.cat(scores).numpy()

    def load_pretrained_embeddings(self, emb_weights):
        expected = self.enc.emb.weight.shape
        if emb_weights.shape != expected:
            log.warning(f'pretrained emb shape {emb_weights.shape} != '
                        f'expected {expected}, skipping')
            return False
        self.enc.emb.weight.data.copy_(emb_weights)
        log.info('loaded pre-trained item embeddings')
        return True


# ---------------------------------------------------------------------------
# loss functions
# ---------------------------------------------------------------------------

def bpr_loss(sp, sn):
    return -torch.log(torch.sigmoid(sp - sn) + 1e-8).mean()


def dyn_loss(all_z, all_t, device):
    parts = []
    for zs, ts in zip(all_z, all_t):
        if not zs or not ts:
            continue
        n = min(len(zs), len(ts))
        z_stack = torch.stack(zs[:n])
        t_stack = torch.stack(ts[:n])
        cos_sim = F.cosine_similarity(z_stack, t_stack, dim=-1)
        parts.append((1.0 - cos_sim).mean())
    if parts:
        return torch.stack(parts).mean()
    return torch.tensor(0., device=device)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def run_eval(model, data, train, n_items, ks=(10, 20), max_u=500):
    model.eval()
    hits = {k: 0. for k in ks}
    ndcgs = {k: 0. for k in ks}
    cnt = 0

    users = list(data.keys())
    if len(users) > max_u:
        users = random.sample(users, max_u)

    for u in tqdm(users, desc='eval', leave=False):
        hist, (gt, gt_t) = data[u]
        if len(hist) < 2:
            continue
        hi = [h[0] for h in hist]
        ht = [h[1] for h in hist]

        # FIX: mask all items in input history, not just train set
        mask_set = {s[0] for s in train.get(u, [])} | {h[0] for h in hist}

        try:
            sc = model.score_all(u, hi, ht, gt_t)
        except RuntimeError:
            continue

        for it in mask_set:
            if it < len(sc):
                sc[it] = -np.inf
        sc[0] = -np.inf

        rank = np.argsort(-sc)
        for k in ks:
            topk = rank[:k]
            if gt in topk:
                hits[k] += 1.
                pos = np.where(topk == gt)[0][0]
                ndcgs[k] += 1. / np.log2(pos + 2)
        cnt += 1

    out = {}
    for k in ks:
        out[f'R@{k}'] = hits[k] / cnt if cnt > 0 else 0.
        out[f'N@{k}'] = ndcgs[k] / cnt if cnt > 0 else 0.
    out['n'] = cnt
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def do_train(args):
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    ds = RecDataset(args.dataset, args.data_dir, args.core)

    model = APTODE(ds.n_users, ds.n_items, args.d, args.h,
                   args.w, args.eta, args.atol, args.rtol).to(dev)

    if args.pretrained_emb and os.path.exists(args.pretrained_emb):
        emb = _safe_load(args.pretrained_emb, dev)
        model.load_pretrained_embeddings(emb)
    elif args.pretrained_emb:
        log.warning(f'pretrained emb file not found: {args.pretrained_emb}')

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f'model params={n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_ds = PairwiseDataset(ds.train, ds.n_items)
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)
    log.info(f'train samples={len(train_ds)} batches/epoch={len(loader)}')

    best_ndcg, stale, best_state = -1., 0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_rec, ep_dyn, nb = 0., 0., 0., 0
        t0 = time.time()

        for batch in tqdm(loader, desc=f'ep{ep}', leave=False):
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
                ep_rec += l_rec.item()
                ep_dyn += l_dyn.item()
                nb += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.warning('OOM, skipped batch')
                else:
                    log.warning(f'batch error: {e}')

        if nb == 0:
            log.warning(f'ep {ep}: all batches failed')
            continue

        dt = time.time() - t0
        log.info(f'ep {ep}/{args.epochs} loss={ep_loss / nb:.4f} '
                 f'rec={ep_rec / nb:.4f} dyn={ep_dyn / nb:.4f} {dt:.0f}s')

        if ep % 5 == 0 or ep == 1:
            n_eval = min(args.eval_users, len(ds.val))
            m = run_eval(model, ds.val, ds.train, ds.n_items, max_u=n_eval)
            log.info(f'  val R@10={m["R@10"]:.4f} N@10={m["N@10"]:.4f} '
                     f'R@20={m["R@20"]:.4f} N@20={m["N@20"]:.4f} n={m["n"]}')

            if m['N@10'] > best_ndcg:
                best_ndcg = m['N@10']
                stale = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                log.info(f'  * best N@10={best_ndcg:.4f}')
            else:
                stale += 5
            if stale >= args.patience:
                log.info('early stop')
                break

    if best_state:
        model.load_state_dict(best_state)

    n_eval = min(args.eval_users, len(ds.test))
    m = run_eval(model, ds.test, ds.train, ds.n_items, max_u=n_eval)
    log.info(f'TEST R@10={m["R@10"]:.4f} N@10={m["N@10"]:.4f} '
             f'R@20={m["R@20"]:.4f} N@20={m["N@20"]:.4f} n={m["n"]}')

    save_path = f'aptode_{args.dataset}.pt'
    torch.save(model.state_dict(), save_path)
    log.info(f'saved to {save_path}')
    return m


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser(description='APT-ODE')
    p.add_argument('--dataset', default='synthetic',
                   help='amazon / steam / ml20m / synthetic')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64, help='embedding dim')
    p.add_argument('--h', type=int, default=128, help='hidden dim')
    p.add_argument('--w', type=int, default=5, help='APT window size')
    p.add_argument('--eta', type=float, default=0.5, help='JSD threshold')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    p.add_argument('--alpha', type=float, default=0.1, help='dyn loss weight')
    p.add_argument('--bs', type=int, default=64, help='batch size')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=20, help='early stop patience')
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--core', type=int, default=5, help='k-core filter')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_users', type=int, default=500)
    p.add_argument('--pretrained_emb', default='',
                   help='path to pretrained embedding .pt file')

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    do_train(cli())
