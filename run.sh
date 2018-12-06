export CUDA_VISIBLE_DEVICES=1
export LC_CTYPE=en_US.UTF-8

nohup python3.6 -u driver/TrainTest.py --config_file config.ptb.cfg --thread 1 --use-cuda > log 2>&1 &
