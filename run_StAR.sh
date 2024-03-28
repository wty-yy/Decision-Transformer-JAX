CUDA_VISIBLE_DEVICES=$1 python StARformer/train.py --path-buffer-root /data/user/wutianyang/dataset/dqn_replay/ $2 --game Breakout
