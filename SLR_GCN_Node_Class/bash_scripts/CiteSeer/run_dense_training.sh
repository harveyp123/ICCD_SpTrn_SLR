cd /home/hop20001/ADMM_SLR_GCN_citation
chmod +x main_dense.py
export EPOCH=200
export DTSTNAME=CiteSeer

export GPU=0
nohup python main_dense.py --epochs ${EPOCH}  --gpus ${GPU} \
      --running-dir /home/hop20001/ADMM_SLR_GCN_citation \
      --dataset ${DTSTNAME} --model-name ${DTSTNAME}_dense \
      --log-name ${DTSTNAME}_dense_train  &