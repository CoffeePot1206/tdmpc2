CUDA_VISIBLE_DEVICES=0 python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
--config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
--config-name cup_none \
>/dev/null 2>&1 <<HERE &
n
HERE

sleep 5s

CUDA_VISIBLE_DEVICES=1 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
--config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
--config-name cup_pos \
>/dev/null 2>&1 <<HERE &
n
HERE

sleep 5s

CUDA_VISIBLE_DEVICES=2 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
--config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
--config-name cup_vel \
>/dev/null 2>&1 <<HERE &
n
HERE

sleep 5s

CUDA_VISIBLE_DEVICES=3 nohup python /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/bc_train.py \
--config-path /home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/config \
--config-name cup_all \
>/dev/null 2>&1 <<HERE &
n
HERE

wait