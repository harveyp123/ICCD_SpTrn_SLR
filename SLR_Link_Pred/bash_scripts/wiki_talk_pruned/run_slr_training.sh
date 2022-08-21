cd /home/hop20001/ADMM_SLR_GNN
chmod +x main_admm.py

export RWEPOCH=10
export RTEPOCH=10
export MOD_IN=wiki_talk_pruned_HID128.ckpt
export DAST=wiki_talk_pruned
export OPT=savlr
export INIT_S=0.01
export RHO=0.01
export GPU=3

export FC1=0.125
export FC2=0.125
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.25
export FC2=0.25
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.375
export FC2=0.375
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.5
export FC2=0.5
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export GPU=4

export FC1=0.625
export FC2=0.625
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.75
export FC2=0.75
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.875
export FC2=0.875
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&


export FC1=0.90625
export FC2=0.90625
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export GPU=5
# export CUDA_VISIBLE_DEVICES=4

export FC1=0.9375
export FC2=0.9375
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.96875
export FC2=0.96875
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.984375
export FC2=0.96875
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&

export FC1=0.9921875
export FC2=0.96875
nohup python main_admm.py --fc1weight ${FC1} --fc2weight ${FC2} --epochs ${RWEPOCH} \
        --retrain-epoch ${RTEPOCH} --gpus ${GPU} --load-model ${MOD_IN} \
        --dataset ${DAST} --log-name ${DAST}_${OPT}_FC1_${FC1}_FC2_${FC2}_checkacc \
        --check-hardprune-acc --optimization ${OPT} --initial_s ${INIT_S} --rho ${RHO}&