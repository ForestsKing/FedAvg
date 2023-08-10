import argparse

import torch

from federated.fed_avg import FedAvg
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='Federated', help='exp name')

    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--download', type=bool, default=False, help='download mnist')

    parser.add_argument('--comm_round', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--shard_size', type=int, default=300)
    parser.add_argument('--num_clients', type=int, default=100, help='K')

    parser.add_argument('--batch_size', type=int, default=32, help='B')
    parser.add_argument('--local_epoch', type=int, default=5, help='E')
    parser.add_argument('--fraction', type=float, default=1.0, help='C')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--devices', type=int, default=0, help='gpu id')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n===================== Args ========================')
    print(args)
    print('===================================================\n')

    setting = '{0}_B{1}_E{2}_C{3:.1f}'.format(args.task, args.batch_size, args.local_epoch, args.fraction)

    setSeed(args.random_seed)

    print('\n>>>>>>>>  initing : {}  <<<<<<<<\n'.format(setting))
    fed_avg = FedAvg(args, setting)

    print('\n>>>>>>>>  training : {}  <<<<<<<<\n'.format(setting))
    fed_avg.fit()

    torch.cuda.empty_cache()

    print('Done!')
