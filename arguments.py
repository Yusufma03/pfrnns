import configargparse
import numpy as np


def parse_args(args=None):

    parser = configargparse.ArgumentParser(default_config_files=[])

    parser.add('-c', '--config', required=True, default='./configs/train.conf',
               is_config_file=True, help='load the config file')

    parser.add_argument('--epochs', type=int, default=800, help='num epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--h', type=int, default=64, help='hidden dim of lstm')
    parser.add_argument('--emb_map', type=int, default=64,
                        help='map embedding dim')
    parser.add_argument('--emb_obs', type=int, default=32,
                        help='observation embedding dim')
    parser.add_argument('--emb_act', type=int, default=32,
                        help='action embedding dim')
    parser.add_argument('--ext_obs', type=int, default=32,
                        help='the size of o(x) in PF-RNNs')
    parser.add_argument('--ext_act', type=int, default=32,
                        help='the size of u(x) in PF-RNNs')

    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('--optim', type=str, default='RMSProp',
                        help='type of optim')
    parser.add_argument('--num_particles', type=int,
                        default=30, help='num of particles')
    parser.add_argument('--sl', type=int, default=100, help='sequence length')
    parser.add_argument('--num_trajs', type=int, default=10000,
                        help='number of trajs')
    parser.add_argument('--resamp_alpha', type=float,
                        default=0.5, help='the soft resampling ratio')
    parser.add_argument('--clip', type=float, default=3.0,
                        help='the grad clip value')
    parser.add_argument('--bp_length', type=int, default=10,
                        help='the truncated bptt length')
    parser.add_argument('--mode', type=str,
                        default='train', help='train or eval')
    parser.add_argument('--model', type=str, default='PFLSTM',
                        help='which model to use for training')
    parser.add_argument('--map_size', type=int, default=10, help='map size')
    parser.add_argument('--gpu', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--bpdecay', type=float, default=0.1,
                        help='the decay along seq for pfrnns')
    parser.add_argument('--obs_num', type=int, default=5, help='observation num')
    parser.add_argument('--h_weight', type=float, default=0.1, help='weight for heading loss')
    parser.add_argument('--l2_weight', type=float, default=1.0, help='weight for l2 loss')
    parser.add_argument('--l1_weight', type=float, default=0.0, help='weight for l1 loss')
    parser.add_argument('--elbo_weight', type=float, default=1.0, help='weight for ELBO loss')

    parser.add_argument('--logs_num', type=int, default=0, help='number of logs folder for your trained model')

    args = parser.parse_args()

    return args
