cd /home/hop20001/ADMM_SLR_GCN_citation
chmod +x main_admm.py

export RWEPOCH=50
export RTEPOCH=50
export MOD_IN=CiteSeer_dense.ckpt
export DAST=CiteSeer
export OPT=savlr
export INIT_S=0.001
export RHO=0.01
export GPU=0

export Sp1=0.1
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.2
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.3
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.4
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.5
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.6
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.7
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.8
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.9
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.92
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.94
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.96
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &

export Sp1=0.98
export Sp2=0
nohup python main_admm.py --conv1linweight ${Sp1} --conv2linweight ${Sp2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_conv1linweight_${Sp1}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO} &