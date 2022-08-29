density=0.5
growth_method=gradient
epoch_num=40
seed=18

python main.py \
    --sparse \
    --seed ${seed} \
    --data_path data/wiki-talk_sparse/ \
    --sparse_init uniform  \
    --multiplier 1 \
    --lr 0.1\
    --gamma 0.1 \
    --density ${density} \
    --update_frequency 1000 \
    --epochs ${epoch_num} \
    --death-rate 0.5 \
    --decay_frequency 30000 \
    --batch-size 128 \
    --growth ${growth_method} \
    --death magnitude

