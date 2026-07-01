Python：>= 3.8  
pip install torch numpy tqdm torchdiffeq    

## 数据下载

Amazon 和 Steam 数据集约 400–500 MB，ML-20M 约 200 MB。建议先用 Amazon 测试。

```bash
mkdir -p data

# Amazon Electronics (~470 MB)
curl -L -o "data/Electronics_5.json.gz" "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"

# Steam (~400 MB)
curl -L -o "data/steam_reviews.json.gz" "https://cseweb.ucsd.edu/~jmcauley/datasets/steam/steam_reviews.json.gz"

# MovieLens-20M (~200 MB)
curl -L -o "data/ratings.csv" "https://files.grouplens.org/datasets/movielens/ml-20m-README.html"
```
> 注：ML-20M 下载地址是说明页，实际数据包 `ml-20m.zip` 可在同站点获取。

### 快速测试（单数据集 Amazon，2 epoch）

```bash
python pretrain.py --dataset amazon --data_dir ./data/ --epochs 5
python apt_ode.py --dataset amazon --data_dir ./data/ --pretrained_emb pretrained_emb_amazon.pt --epochs 2 --eval_users 100 --bs 512
```

---

project/
├── apt_ode.py         # 主模型：APT-ODE 训练与评估
├── pretrain.py        # BPR-MF 预训练 item embeddings
├── ablation.py        # 消融实验（主消融 + 设计选择消融）
├── sensitivity.py     # 超参数敏感性分析（三数据集）
├── efficiency.py      # 效率测量（训练时间 / 推理延迟 / GPU 内存）
├── plot_sensitivity.py # 生成图3 参数敏感性分析 PDF
└── data/
    ├── Electronics_5.json.gz     
    ├── steam_reviews.json.gz     
    └── ratings.csv    

## 真实数据集完整运行流程

### 1. 预训练 BPR-MF embeddings（每个数据集跑一次）

```bash
python pretrain.py --dataset amazon --data_dir ./data/ --epochs 20
python pretrain.py --dataset steam   --data_dir ./data/ --epochs 20
python pretrain.py --dataset ml20m   --data_dir ./data/ --epochs 20
```
输出：`pretrained_emb_amazon.pt` / `pretrained_emb_steam.pt` / `pretrained_emb_ml20m.pt`

### 2. 训练 APT-ODE（5-seed，获取 mean±std）

```bash
python apt_ode.py --dataset amazon --data_dir ./data/ --pretrained_emb pretrained_emb_amazon.pt --n_seeds 5
python apt_ode.py --dataset steam  --data_dir ./data/ --pretrained_emb pretrained_emb_steam.pt  --n_seeds 5
python apt_ode.py --dataset ml20m  --data_dir ./data/ --pretrained_emb pretrained_emb_ml20m.pt  --n_seeds 5
```
输出：每个数据集训练结束后打印 5-seed mean±std，保存模型 `aptode_<dataset>.pt`

### 3. 消融实验（Table 4 + Table 5）

```bash
# 每个数据集跑全部消融（主消融 + 设计选择消融）
python ablation.py --dataset amazon --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5
python ablation.py --dataset steam  --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5
python ablation.py --dataset ml20m  --data_dir ./data/ --ablation all --epochs 15 --n_seeds 5

# 或分开跑：
python ablation.py --dataset amazon --data_dir ./data/ --ablation main --n_seeds 5    # Table 4
python ablation.py --dataset amazon --data_dir ./data/ --ablation design --n_seeds 5  # Table 5
```

### 4. 超参数敏感性分析（图3 / RQ3）

```bash
# 三数据集一次性跑完
python sensitivity.py --datasets all --data_dir ./data/ --epochs 10

# 或指定单个数据集
python sensitivity.py --datasets amazon --data_dir ./data/ --epochs 10
python sensitivity.py --datasets steam   --data_dir ./data/ --epochs 10
python sensitivity.py --datasets ml20m   --data_dir ./data/ --epochs 10
```
输出：每个数据集的 delta 和 w 敏感性结果，以及跨数据集最优值汇总

### 5. 效率测量（Table 6 / RQ5）

```bash
python efficiency.py --dataset ml20m --data_dir ./data/ --pretrained_emb pretrained_emb_ml20m.pt
```
输出：训练时间 (s/epoch)、推理延迟 (ms/user)、GPU 内存 (GB)

### 6. 生成图3（参数敏感性分析 PDF）

```bash
python plot_sensitivity.py
```
输出：`Parameter_Sensitivity.pdf`（三数据集 δ 和 w 敏感性曲线）

---

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | amazon | amazon / steam / ml20m |
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
| `--bs` | 2048 | 批大小（与论文一致） |
| `--core` | 5 | k-core 过滤阈值 |
| `--eval_users` | 500 | 评估用户数上限 |
| `--atol` / `--rtol` | 1e-5 | ODE solver 容差 |
| `--ablation` | main | 消融类型：main / design / all |
| `--datasets` | all | 敏感性分析数据集：amazon / steam / ml20m / all |
