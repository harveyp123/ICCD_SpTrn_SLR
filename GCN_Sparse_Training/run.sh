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



# density_list=(0.2 0.1 0.05 0.02 0.01)
# density_list=(0.875 0.75 0.625 0.5 0.375 0.25 0.125 0.09375 0.0625 0.03125 0.015625 0.0078125)
# growth_method=gradient
# epoch_num=40
# seed=18
# # density=0.05

# let list_length=${#density_list[@]}-1
# for index in $(seq 0 ${list_length})
# do
#     density=${density_list[index]}
#     bash sparse_train_5.sh $@ \
#         --sparse \
#         --seed ${seed} \
#         --data_path data/ia-email_dense/ \
#         --sparse_init uniform  \
#         --multiplier 1 \
#         --lr 0.1\
#         --gamma 0.1 \
#         --density ${density} \
#         --update_frequency 1000 \
#         --epochs ${epoch_num} \
#         --death-rate 0.5 \
#         --decay_frequency 30000 \
#         --batch-size 128 \
#         --growth ${growth_method} \
#         --theta 1e-2 \
#         --theta_decay_freq 1000 \
#         --factor 0.9 \
#         --epsilon 1 \
#         --death magnitude \
#         --redistribution none 
# done
