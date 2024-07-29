NUM_DEMOS=100
SEED=1
NUM_EPOCH=300
START_FROM=0
LR=2e-4
TASK="cup-harder"

seed_list=(2 3)
# num_demo_list=(100 200 500)
num_demo_list=(50 25 10 100 200 500 1000)
state_list=(none pos vel all)
# state_list=(debug)

run(){
CUDA_VISIBLE_DEVICES=$1 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
--config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
--config-name cup_$2 \
task=$TASK \
dataset="/cache1/kuangfuhang/tdmpc2/datasets/$TASK/rgb-${3}.hdf5" \
seed=$4 \
num_epochs=$NUM_EPOCH \
start_from=$START_FROM \
batch_size=$5 \
learning_rate.initial=$6 \
>/dev/null 2>&1 <<HERE &
n
HERE
}

# for SEED in ${seed_list[*]}; do
for NUM_DEMOS in ${num_demo_list[*]}; do
gpu=0
for STATE in ${state_list[*]}; do

# BS=256
# VBS=64
# LR=2e-4

if [ $NUM_DEMOS -ge 50 ]; then
    # BS=256
    # let VBS=32
    # let LR=0.0002
    # echo $NUM_DEMOS, $BS, $VBS, $LR
    run $gpu $STATE $NUM_DEMOS $SEED 256 2e-4
elif [ $NUM_DEMOS -eq 25 ]; then
    # BS=128
    # let VBS=16
    # let LR=0.0001
    # echo $NUM_DEMOS, $BS, $VBS, $LR
    run $gpu $STATE $NUM_DEMOS $SEED 128 1e-4
elif [ $NUM_DEMOS -eq 10 ]; then
    # let BS=64
    # let VBS=4
    # let LR=0.00005
    run $gpu $STATE $NUM_DEMOS $SEED 64 5e-5
else
    echo "nothing"
fi

# echo $NUM_DEMOS, $BS, $VBS, $LR

let gpu=gpu+1
done
wait
done
# done

exit 0


# CUDA_VISIBLE_DEVICES=0 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
# --config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
# --config-name cup_none \
# dataset="/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgb-${NUM_DEMOS}.hdf5" \
# seed=$SEED \
# >/dev/null 2>&1 <<HERE &
# n
# HERE

# CUDA_VISIBLE_DEVICES=1 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
# --config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
# --config-name cup_pos \
# dataset="/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgb-${NUM_DEMOS}.hdf5" \
# seed=$SEED \
# >/dev/null 2>&1 <<HERE &
# n
# HERE

# CUDA_VISIBLE_DEVICES=2 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
# --config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
# --config-name cup_vel \
# dataset="/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgb-${NUM_DEMOS}.hdf5" \
# seed=$SEED \
# >/dev/null 2>&1 <<HERE &
# n
# HERE

# CUDA_VISIBLE_DEVICES=3 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
# --config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
# --config-name cup_all \
# dataset="/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgb-${NUM_DEMOS}.hdf5" \
# seed=$SEED \
# >/dev/null 2>&1 <<HERE &
# n
# HERE

# wait