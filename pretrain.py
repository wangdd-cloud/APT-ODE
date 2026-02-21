"""
BPR-MF pre-training for APT-ODE item embeddings.
Outputs a .pt file that can be loaded by apt_ode.py via --pretrained_emb.
"""

import os, sys, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
from apt_ode import RecDataset, set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, d):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        self.item_emb.weight.data[0].zero_()

    def forward(self, uids, pids, nids):
        u = self.user_emb(uids)
        p = self.item_emb(pids)
        n = self.item_emb(nids)
        sp = (u * p).sum(-1)
        sn = (u * n).sum(-1)
        return -torch.log(torch.sigmoid(sp - sn) + 1e-8).mean()


def build_pairs(train, n_items):
    pairs = []
    for u, seq in train.items():
        pos_set = {s[0] for s in seq}
        for s in seq:
            neg = random.randint(1, n_items - 1)
            while neg in pos_set:
                neg = random.randint(1, n_items - 1)
            pairs.append((u, s[0], neg))
    return pairs


def pretrain(args):
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'device={dev}')

    ds = RecDataset(args.dataset, args.data_dir, args.core)
    model = BPRMF(ds.n_users, ds.n_items, args.d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pairs = build_pairs(ds.train, ds.n_items)
    log.info(f'training pairs={len(pairs)}')

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        total_loss, nb = 0., 0

        for i in range(0, len(pairs), args.bs):
            batch = pairs[i:i + args.bs]
            u_t = torch.LongTensor([p[0] for p in batch]).to(dev)
            p_t = torch.LongTensor([p[1] for p in batch]).to(dev)
            n_t = torch.LongTensor([p[2] for p in batch]).to(dev)

            loss = model(u_t, p_t, n_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            nb += 1

        if ep % 5 == 0 or ep == 1:
            log.info(f'ep {ep}/{args.epochs} loss={total_loss / nb:.4f}')

    out_path = f'pretrained_emb_{args.dataset}.pt'
    torch.save(model.item_emb.weight.data.cpu(), out_path)
    log.info(f'saved embeddings to {out_path}')
    return out_path


def cli():
    p = argparse.ArgumentParser(description='BPR-MF pre-training')
    p.add_argument('--dataset', default='synthetic')
    p.add_argument('--data_dir', default='./data/')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--core', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    if 'ipykernel' in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == '__main__':
    pretrain(cli())
