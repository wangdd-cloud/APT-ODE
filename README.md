Python：>= 3.8  
pip install torch numpy tqdm torchdiffeq    

```
APT-ODE/
├── apt_ode.py          # 主模型训练与评估
├── pretrain.py         # BPR-MF 预训练
├── ablation.py         # 消融实验
├── sensitivity.py      # 超参数敏感性分析
├── efficiency.py       # 效率测量
└── data/               # 数据集（需自行下载）
```

## 真实数据集完整运行流程

### 1. 预训练 BPR-MF embeddings

```bash
python pretrain.py --dataset amazon --data_dir ./data/ --epochs 20
python pretrain.py --dataset steam   --data_dir ./data/ --epochs 20
python pretrain.py --dataset ml20m   --data_dir ./data/ --epochs 20
```
输出：`pretrained_emb_amazon.pt` / `pretrained_emb_steam.pt` / `pretrained_emb_ml20m.pt`

### 2. 训练 APT-ODE

```bash
python apt_ode.py --dataset amazon --data_dir ./data/ --pretrained_emb pretrained_emb_amazon.pt --n_seeds 5
python apt_ode.py --dataset steam  --data_dir ./data/ --pretrained_emb pretrained_emb_steam.pt  --n_seeds 5
python apt_ode.py --dataset ml20m  --data_dir ./data/ --pretrained_emb pretrained_emb_ml20m.pt  --n_seeds 5
```
输出：每个数据集训练结束后打印 5-seed mean±std，保存模型 `aptode_<dataset>.pt`

### 3. 消融实验

```bash
python ablation.py --dataset amazon --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5
python ablation.py --dataset steam  --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5
python ablation.py --dataset ml20m  --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5
```

### 4. 超参数敏感性分析

```bash
# 三数据集一次性跑完
python sensitivity.py --datasets all --data_dir ./data/ --epochs 10

# 或指定单个数据集
python sensitivity.py --datasets amazon --data_dir ./data/ --epochs 10
python sensitivity.py --datasets steam   --data_dir ./data/ --epochs 10
python sensitivity.py --datasets ml20m   --data_dir ./data/ --epochs 10
```
输出：每个数据集的 delta 和 w 敏感性结果，以及跨数据集最优值汇总

### 5. 效率测量

```bash
python efficiency.py --dataset ml20m --data_dir ./data/ --pretrained_emb pretrained_emb_ml20m.pt
```
输出：训练时间 (s/epoch)、推理延迟 (ms/user)、GPU 内存 (GB)

---

## 快速测试

```bash
python apt_ode.py --dataset synthetic --epochs 2 --eval_users 50 --bs 32
python apt_ode.py --dataset synthetic --epochs 5 --n_seeds 3 --eval_users 50 --bs 32
python ablation.py --dataset synthetic --epochs 5 --ablation main --eval_users 50 --bs 32
python ablation.py --dataset synthetic --epochs 5 --ablation design --eval_users 50 --bs 32
python sensitivity.py --datasets synthetic --epochs 5 --eval_users 50 --bs 32
python efficiency.py --dataset synthetic --bs 32
```

---

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | synthetic | amazon / steam / ml20m / synthetic |
| `--data_dir` | ./data/ | 数据文件目录 |
| `--pretrained_emb` | (空) | BPR-MF 预训练 embedding 路径 |
| `--delta` | 0.5 | JSD 边界检测阈值 |
| `--w` | 5 | APT 滑动窗口大小 |
| `--n_seeds` | 1 | 随机种子数（≥2 输出 mean±std） |
| `--seed` | 42 | 起始随机种子 |
| `--d` | 64 | embedding 维度 |
| `--h` | 128 | 向量场 MLP 隐藏层维度 |
| `--lr` | 1e-3 | 学习率 |
| `--wd` | 1e-4 | L2 正则化系数 |
| `--alpha` | 0.1 | 轨迹对齐损失权重 |
| `--epochs` | 50 | 最大训练轮数 |
| `--patience` | 20 | 早停轮数 |
| `--bs` | 2048 | 批大小 |
| `--core` | 5 | k-core 过滤阈值 |
| `--eval_users` | 500 | 评估用户数上限 |
| `--atol` / `--rtol` | 1e-5 | ODE solver 容差 |
| `--ablation` | main | 消融类型：main / design / all |
| `--datasets` | all | 敏感性分析数据集：amazon,steam,ml20m,synthetic 或 all |
