import torch
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
import numpy as np
from model import Localizer
from arguments import parse_args
import os
from torch.utils.tensorboard import SummaryWriter
import logging
import pickle
import time

if not os.path.isdir('eval'):
    os.mkdir('eval')

def get_data_name(args, train):
    """
    get the dataset name

    :param args: experiment args
    :param train: train / eval
    :return: fname: the name of the file
    """
    # number of trajs
    num_trajs = args.num_trajs
    # number of trajs
    traj_len = args.sl 

    mode = 'train' if train else 'eval'

    fname = '{}_data_trajs{}_sl{}.pkl'.format(mode, num_trajs, traj_len)
    return fname


def get_logger():
    root = './eval'
    existings = os.listdir(root)
    cnt = str(len(existings))
    logger = SummaryWriter(os.path.join(root, cnt, 'tflogs'))

    return logger, cnt


def save_args(args, run_id):
    ret = vars(args)
    path = os.path.join('eval', run_id, 'args.conf')
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


def set_logging(cnt):
    file_path = os.path.join("./eval", cnt, "particle_pred.log")
    logging.basicConfig(filename=file_path, level=logging.DEBUG)


def get_checkpoint(args):
    try:
        model_checkpoint = torch.load(os.path.join(os.getcwd(), 'logs', str(args.logs_num), 'models', 'model_best'))
        optimizer_checkpoint = torch.load(os.path.join(os.getcwd(), 'logs', str(args.logs_num), 'models', 'optim_best'))
    except:
        print("\n[Error] Please make sure you have trained the model using main.py. ")
        print("And set the correct model path. \n")
    
    return model_checkpoint, optimizer_checkpoint


def evaluate(args, logger, run_id):
    model = get_model(args)
    optimizer = get_optim(args, model)

    model_checkpoint, optimizer_checkpoint = get_checkpoint(args)
    model.load_state_dict(model_checkpoint)
    optimizer.load_state_dict(optimizer_checkpoint)
    model.eval()

    train_data, eval_data = get_data(args)
    eval_dataset = LocalizationDataset(eval_data)

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             num_workers=8, pin_memory=True, shuffle=False)

    from timer import Timer
    infer_timer = Timer("Inference")

    cnt = 0
    from tqdm import tqdm

    for epoch in tqdm(range(args.epochs)):
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
                infer_timer.start() # start the timer
                loss, log_loss, particle_pred = model.step(
                    env_map, obs, action, pos, args)
                infer_timer.stop() # pause the timer

                eval_loss_all.append(loss.to('cpu').detach().numpy())
                eval_loss_last.append(log_loss.to('cpu').detach().numpy())

        log_eval_last = np.mean(eval_loss_last)
        log_eval_all = np.mean(eval_loss_all)
        logger.add_scalar('eval/loss_last', log_eval_last, cnt)
        logger.add_scalar('eval/loss', log_eval_all, cnt)

        logging.info(particle_pred.size())
        logging.info("time elapse average %f" % (infer_timer.average))
        logging.info("time elapse total %f" % (infer_timer.total))
        logging.info("time elapses " + str(infer_timer.time_log))
        logging.info("================ seperate line =================")

        #### save particle_pred tensor for plot_particle.py ####
        fname = os.path.join('eval', run_id, 'particle_pred')
        torch.save(particle_pred, fname)

        cnt += 1


if __name__ == "__main__":
    args = parse_args()
    loggers, run_id = get_logger()
    save_args(args, run_id)
    set_logging(run_id)
    evaluate(args, loggers, run_id)

