export DTSTNAME=Cora
export GPU=0
export density=0.1


python main.py \
        --sparse \
        --seed 18 \
        --dataset ${DTSTNAME} \
        --model-name ${DTSTNAME}_dense \
        --log-name ${DTSTNAME}_dense_train \
        --gpus ${GPU} \
        --sparse_init ERK  \
        --multiplier 1 \
        --lr 0.01\
        --gamma 0.1 \
        --density ${density} \
        --update_frequency 50 \
        --epochs 300 \
        --death-rate 0.5 \
        --decay_frequency 50 \
        --batch_size 128 \
        --growth gradient \
        --death magnitude \
        --redistribution none 
