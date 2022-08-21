cd /home/hop20001/ADMM_SLR_GNN_Node_class
chmod +x main_admm.py

export RWEPOCH=10
export RTEPOCH=10
export MOD_IN=brain_dense.ckpt
export DAST=brain_dense
export OPT=admm
export INIT_S=0.02
export RHO=0.02
export GPU=3

export Sp=0.125
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export Sp=0.25
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export Sp=0.375
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export Sp=0.5
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export GPU=4

export Sp=0.625
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export Sp=0.75
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export Sp=0.875
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&

export GPU=5
# export CUDA_VISIBLE_DEVICES=4

export Sp=0.90625
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}\
        --running-dir /home/hop20001/ADMM_SLR_GNN_Node_class&



export Sp=0.9375
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export Sp=0.96875
nohup python main_admm.py --sparsity ${Sp} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_Sp_${Sp}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&
