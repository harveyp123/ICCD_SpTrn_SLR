lr=0.1
growth_method=gradient
epoch_num=50
seed=18
density=0.05

CUDA_VISIBLE_DEVICES=0 python main.py \
        --sparse \
        --seed ${seed} \
        --sparse_init uniform  \
        --multiplier 1 \
        --lr ${lr} \
        --gamma 0.1 \
        --density ${density} \
        --update_frequency 500 \
        --epochs ${epoch_num} \
        --death-rate 0.5 \
        --decay_frequency 30000 \
        --batch-size 128 \
        --growth ${growth_method} \
        --theta 1e-2 \
        --theta_decay_freq 1000 \
        --factor 0.9 \
        --epsilon 1 \
        --death magnitude \
        --redistribution none 
