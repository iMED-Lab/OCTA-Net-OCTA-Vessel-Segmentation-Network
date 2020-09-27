Coarse Stage:
    python front_main.py
Fine Stage:
    python fusion_main.py
python test_time.py

train: --mode train
test: --mode test

SCC (ROSE-1): --dataset rose --data_dir ../../data/ROSE-1/SCC
DCC (ROSE-1): --dataset rose --data_dir ../../data/ROSE-1/DCC
WRCC (ROSE-1): --dataset rose --data_dir ../../data/ROSE-1/WRCC
ROSE-2: --dataset cria --data_dir ../../data/ROSE-2
