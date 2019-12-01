import torch
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
import numpy as np
from model import Localizer
from arguments import parse_args
import os
from torch.utils.tensorboard import SummaryWriter


if not os.path.isdir('logs'):
    os.mkdir('logs')


def get_data_name(args, train):
    """
    get the dataset name

    :param args: experiment args
    :param train: train / eval
    :return: fname: the name of the file
    """
    num_trajs = args.num_trajs
    traj_len = args.sl

    mode = 'train' if train else 'eval'

    fname = '{}_data_trajs{}_sl{}.pkl'.format(mode, num_trajs, traj_len)
    return fname


def get_logger():
    root = './logs'
    existings = os.listdir(root)
    cnt = str(len(existings))
    logger = SummaryWriter(os.path.join(root, cnt, 'tflogs'))

    return logger, cnt


def save_args(args, run_id):
    ret = vars(args)
    path = os.path.join('logs', run_id, 'args.conf')
    import json
    with open(path, 'w') as fout:
        json.dump(ret, fout)


def get_optim(args, model):
    if args.optim == 'RMSProp':
        optim = torch.optim.RMSprop(
            model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optim


def get_model(args):
    model = Localizer(args)
    if torch.cuda.is_available() and args.gpu:
        model = model.to('cuda')
    return model


def get_data(args):
    """
    get the localization dataset, both train and eval

    :param args: experiment args
    :return: train_data: the localization training data
             eval_data: the localization evaluation data
    """
    train_fname = get_data_name(args, True)
    eval_fname = get_data_name(args, False)
    import pickle

    if not os.path.isdir('data'):
        os.mkdir('data')

    try:
        with open(os.path.join('data', train_fname), 'rb') as fin:
            train_data = pickle.load(fin)
        with open(os.path.join('data', eval_fname), 'rb') as fin:
            eval_data = pickle.load(fin)
    except:
        import data_utils
        print("Load data failed, generating training data")

        train_data = data_utils.gen_data(args.num_trajs, args.sl)
        eval_data = data_utils.gen_data(args.num_trajs // 10, args.sl)

        with open(os.path.join('data', train_fname), 'wb') as fout:
            pickle.dump(train_data, fout, pickle.HIGHEST_PROTOCOL)
        print("train data generated")

        with open(os.path.join('data', eval_fname), 'wb') as fout:
            pickle.dump(eval_data, fout, pickle.HIGHEST_PROTOCOL)
        print("eval data generated")

    return train_data, eval_data


def train(args, logger, run_id):
    model = get_model(args)
    optimizer = get_optim(args, model)

    train_data, eval_data = get_data(args)
    train_dataset = LocalizationDataset(train_data)
    eval_dataset = LocalizationDataset(eval_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=8, pin_memory=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             num_workers=8, pin_memory=True, shuffle=False)

    os.mkdir(os.path.join('logs', run_id, 'models'))

    cnt = 0
    best_eval = 1000
    from tqdm import tqdm

    for epoch in tqdm(range(args.epochs)):
        model.train()

        for iteration, data in enumerate(train_loader):
            cnt = cnt + 1

            env_map, obs, pos, action = data

            if torch.cuda.is_available() and args.gpu:
                env_map = env_map.to('cuda')
                obs = obs.to('cuda')
                pos = pos.to('cuda')
                action = action.to('cuda')

            model.zero_grad()
            loss, log_loss, particle_pred = model.step(
                env_map, obs, action, pos, args)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if iteration % 50:
                loss_last = log_loss.to('cpu').detach().numpy()
                loss_all = loss.to('cpu').detach().numpy()

                logger.add_scalar('train/loss_last', loss_last, cnt)
                logger.add_scalar('train/loss', loss_all, cnt)

        model.eval()
        eval_loss_all = []
        eval_loss_last = []
        with torch.no_grad():
            for iteration, data in enumerate(eval_loader):
                env_map, obs, pos, action = data

                if torch.cuda.is_available() and args.gpu:
                    env_map = env_map.to('cuda')
                    obs = obs.to('cuda')
                    pos = pos.to('cuda')
                    action = action.to('cuda')

                model.zero_grad()
                loss, log_loss, particle_pred = model.step(
                    env_map, obs, action, pos, args)

                eval_loss_all.append(loss.to('cpu').detach().numpy())
                eval_loss_last.append(log_loss.to('cpu').detach().numpy())

        log_eval_last = np.mean(eval_loss_last)
        log_eval_all = np.mean(eval_loss_all)
        logger.add_scalar('eval/loss_last', log_eval_last, cnt)
        logger.add_scalar('eval/loss', log_eval_all, cnt)

        if log_eval_last < best_eval:
            best_eval = log_eval_last
            torch.save(model.state_dict(), os.path.join(
                'logs', run_id, 'models', 'model_best'))
            torch.save(optimizer.state_dict(), os.path.join(
                'logs', run_id, 'models', 'optim_best'))

    torch.save(model.state_dict(), os.path.join(
        'logs', run_id, 'models', 'model_final'))
    torch.save(optimizer.state_dict(), os.path.join(
        'logs', run_id, 'models', 'optim_final'))


if __name__ == "__main__":
    args = parse_args()
    logger, run_id = get_logger()
    save_args(args, run_id)
    train(args, logger, run_id)
