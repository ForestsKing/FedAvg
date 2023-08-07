import argparse
import time

import torch

from exp.exp import EXP
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='Centralized')

    parser.add_argument('--download', type=bool, default=False, help='download mnist')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--devices', type=int, default=0, help='gpu id')

    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n===================== Args ========================')
    print(args)
    print('===================================================\n')

    time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_name = f"{args.task}_{time_now}"

    setting = f"{exp_name}"

    setSeed(args.random_seed)

    print('\n>>>>>>>>  initing : {}  <<<<<<<<\n'.format(setting))
    exp = EXP(args, setting)

    print('\n>>>>>>>>  training : {}  <<<<<<<<\n'.format(setting))
    exp.train()

    print('\n>>>>>>>>  testing : {}  <<<<<<<<\n'.format(setting))
    exp.test()

    torch.cuda.empty_cache()

    print('Done!')
