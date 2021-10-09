# @Author:Xie Ningwei
# @Date:2021-10-09 14:58:59
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-10-09 16:06:42
import argparse
import logging
import os
from collections import OrderedDict, defaultdict
# torch
import torch
from torch import nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# self defined
from models.dgnn import Model
from utils.mylogger import create_logger
from feeders.feeder import Feeder
from utils.trainer import DGNN_Trainer


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_param_groups(model):
    param_groups = defaultdict(list)
    for name, params in model.named_parameters():
        if ('source_M' in name) or ('target_M' in name):
            param_groups['graph'].append(params)
        else:
            param_groups['other'].append(params)


    # NOTE: Different parameter groups should have different learning behaviour
    optim_param_groups = {
        'graph': {'params': param_groups['graph']},
        'other': {'params': param_groups['other']}
    }
    return param_groups, optim_param_groups


def main(args):
    # initialize logger
    logger = create_logger(
        filename=args.log_file,
        logger_prefix=__file__
    )

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_class=args.class_num, num_point=args.joint_num).to(device)

    # initialize criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Load weights
    if os.path.exists(args.resume) and args.resume.endswith('.pth'):
        logging.info("Loading the checkpoint from {}".format(args.resume))
        check = torch.load(args.resume, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(check['model'])
            args.start_epoch = check['epoch']
        except:
            print('Fail to load checkpoint {}'.format(args.resume))

    # initialize optimizer
    param_groups, optim_param_groups = load_param_groups(model)
    p_groups = list(optim_param_groups.values())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            p_groups,
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            p_groups,
            lr=args.base_lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optimizer))

    # initialize scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=0.1)

    # initialize dataloader
    train_loader = DataLoader(
        dataset = Feeder(args.train_joint_file, args.train_bone_file, args.test_label_file),
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True,
                worker_init_fn=init_seed
    )
    test_loader = DataLoader(
        dataset=Feeder(args.test_joint_file, args.test_bone_file, args.test_label_file),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        worker_init_fn=init_seed
    )

    trainer = DGNN_Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, train_loader=train_loader, valid_loader=test_loader, test_loader=test_loader,
        start_epoch=args.start_epoch, end_epoch=args.end_epoch, logger=logger, model_save_dir=args.checkpoint_dir,
        save_interval=args.save_interval, param_groups=param_groups, freeze_graph_until=args.freeze_graph_until
    )
    if args.mode == 'validate':
        trainer.validate(model_path=args.resume)
    else:
        trainer.train()
        trainer.validate(model_path=os.path.join(args.checkpoint_dir,"model_best.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition Based on DGNN')
    # general
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--device', default=[0], type=list)

    # optimizer
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--lr_patience', default=5, type=int)
    parser.add_argument('--step', default=[30,60,90], type=list)

    # epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=150, type=int)
    parser.add_argument('--freeze_graph_until', default=20, type=int)

    # batch_size
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=32, type=int)

    # checkpoints
    parser.add_argument('--checkpoint_dir', default='.\\checkpoints\\', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--log_file', default='.\\checkpoints\\train.log', type=str)
    parser.add_argument('--save_interval', default=20, type=int)

    # dataset
    parser.add_argument('--class_num', default=14, type=int)
    parser.add_argument('--joint_num', default=22, type=int)
    parser.add_argument(
        '--train_joint_file',
        default='..\\data\\train_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_bone_file',
        default='..\\data\\train_bone_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_label_file',
        default='..\\data\\train_label.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_joint_file',
        default='..\\data\\test_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_bone_file',
        default='..\\data\\test_bone_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_label_file',
        default='..\\data\\test_label.npy',
        type=str,
        metavar='PATH'
    )

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)