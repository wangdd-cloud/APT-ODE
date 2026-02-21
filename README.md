Python：>= 3.8  
pip install torch numpy tqdm torchdiffeq    
project/
├── apt_ode.py  
├── pretrain.py  
├── ablation.py  
├── sensitivity.py  
└── data/  
    ├── Electronics_5.json.gz     
    ├── steam_reviews.json.gz     
    └── ratings.csv    
python pretrain.py   
python apt_ode.py   
python ablation.py   
python sensitivity.py   
python pretrain.py --dataset amazon --data_dir ./data/  
python apt_ode.py --dataset amazon --data_dir ./data/ --pretrained_emb pretrained_emb_amazon.pt  
python ablation.py --dataset amazon --data_dir ./data/  
python sensitivity.py --dataset amazon --data_dir ./data/  
