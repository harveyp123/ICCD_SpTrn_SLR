cd /home/hop20001/ADMM_SLR_GNN_Node_class
chmod +x main_dense.py
export EPOCH=20
export LR_DC=0.99
export DTSTNAME=brain_sparse
export GPU=2
python main_dense.py --num-epochs ${EPOCH} --batch-size 512 --learning-rate 0.005\
      --gpus ${GPU} --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME} \
      --log-name ${DTSTNAME}_train  --lr-decay ${LR_DC} --learning-rate 0.004