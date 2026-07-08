Python：>= 3.8  
pip install torch numpy tqdm torchdiffeq    

---

project/
├── apt_ode.py         # 主模型：APT-ODE 训练与评估
├── pretrain.py        # BPR-MF 预训练 item embeddings
├── ablation.py        # 消融实验（主消融 + 设计选择消融）
├── sensitivity.py     # 超参数敏感性分析（三数据集）
├── efficiency.py      # 效率测量（训练时间 / 推理延迟 / GPU 内存）
├── analysis.py        # 可解释性分析（RQ4：segment 统计、边界分析、品类转换、短序列评估）
├── data/
     
主要参数

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
| `--model_path` | (空) | analysis.py 专用：训练好的 APT-ODE checkpoint 路径 |

## 运行示例

```bash
# 主实验 (5 种子)
python apt_ode.py --dataset amazon --data_dir ./data/ --n_seeds 5

# 消融实验
python ablation.py --dataset amazon --data_dir ./data/ --ablation all --n_seeds 5

# 敏感性分析
python sensitivity.py --datasets all --data_dir ./data/

# 效率测量
python efficiency.py --dataset ml20m --data_dir ./data/

# 可解释性分析 (需要先训练模型)
python analysis.py --dataset amazon --data_dir ./data/ --model_path aptode_amazon.pt
```
