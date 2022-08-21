cd /home/hop20001/ADMM_SLR_GNN
chmod +x main_dense.py
export EPOCH=20
export LR_DC=0.98
export DTSTNAME=stackoverflow_dense
export GPU=2
export HID=4
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=8
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=12
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=16
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &

export GPU=3

export HID=32
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=48
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=64
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &

export GPU=4

export HID=80
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=96
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &
export HID=112
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &

export GPU=2
export HID=128
nohup python main_dense.py --input-size 16 --hidden-size ${HID} --num-classes 1 --num-epochs ${EPOCH} \
      --batch-size 1024 --learning-rate 0.05 --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_HID${HID} --lr-decay ${LR_DC} \
      --log-name ${DTSTNAME}_train_HID${HID} &