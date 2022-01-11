# @Author:Xie Ningwei
# @Date:2021-11-24 11:07:35
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-11-26 16:27:19
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
from models.dgnn import DGNN
from models.dg_sta import DG_STA
from models.dgnn_att import DGNN_ATT
from models.twostr_dgnn import TwoStreamDGNN
from utils.mylogger import create_logger
from feeders.feeder import DGNN_Feeder, DG_STA_Feeder, TwoStreamDGNN_Feeder
from utils.trainer import DGNN_Trainer, DG_STA_Trainer, TwoStreamDGNN_Trainer



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


def train_DGNN(args):
    # initialize logger
    logger = create_logger(
        filename=args.log_file,
        logger_prefix=__file__
    )

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGNN(num_class=args.class_num, num_point=args.joint_num).to(device)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # initialize dataloader
    train_loader = DataLoader(
        dataset=DGNN_Feeder(args.train_joint_file, args.train_bone_file, args.train_label_file),
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True
    )
    valid_loader = DataLoader(
        dataset=DGNN_Feeder(args.test_joint_file, args.test_bone_file, args.test_label_file),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=DGNN_Feeder(args.test_joint_file, args.test_bone_file, args.test_label_file),
        batch_size = int(2800*0.3),
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    trainer = DGNN_Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        start_epoch=args.start_epoch, end_epoch=args.end_epoch, logger=logger, model_save_dir=args.checkpoint_dir,
        save_interval=args.save_interval, param_groups=param_groups, freeze_graph_until=args.freeze_graph_until, class_num=args.class_num
    )

    trainer.train()
    trainer.validate()


def train_TwoStreamDGNN(args):
    # initialize logger
    logger = create_logger(
        filename=args.log_file,
        logger_prefix=__file__
    )

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoStreamDGNN(num_class=args.class_num, num_point=args.joint_num).to(device)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # initialize dataloader
    train_loader = DataLoader(
        dataset=TwoStreamDGNN_Feeder(args.train_joint_file, args.train_bone_file, args.train_joint_motion_file, args.train_bone_motion_file, args.train_label_file),
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True
    )
    valid_loader = DataLoader(
        dataset=TwoStreamDGNN_Feeder(args.test_joint_file, args.test_bone_file, args.test_joint_motion_file, args.test_bone_motion_file,args.test_label_file),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=TwoStreamDGNN_Feeder(args.test_joint_file, args.test_bone_file, args.test_joint_motion_file, args.test_bone_motion_file,args.test_label_file),
        batch_size = int(2800*0.2),
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    trainer = TwoStreamDGNN_Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        start_epoch=args.start_epoch, end_epoch=args.end_epoch, logger=logger, model_save_dir=args.checkpoint_dir,
        save_interval=args.save_interval, param_groups=param_groups, freeze_graph_until=args.freeze_graph_until, class_num=args.class_num
    )

    #trainer.train()
    trainer.validate()


def parse_args():
    parser = argparse.ArgumentParser(description='Skeleton-based Hand Gesture Recognition')
    # general
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--device', default=[0], type=list)

    # hyper-parameters
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--lr_patience', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=int)
    parser.add_argument('--step', default=[30,60,90], type=list)

    # epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=150, type=int)
    parser.add_argument('--freeze_graph_until', default=10, type=int)

    # batch_size
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=32, type=int)

    # checkpoints
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', type=str, metavar='PATH')
    parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str)
    parser.add_argument('--log_file', default='./checkpoints/train.log', type=str, metavar='PATH')
    parser.add_argument('--save_interval', default=20, type=int)

    # dataset
    parser.add_argument('--class_num', default=14, type=int)
    parser.add_argument('--joint_num', default=22, type=int)
    parser.add_argument('--data_rootdir', default='./data/', type=str, metavar='PATH')
    parser.add_argument(
        '--train_joint_file',
        default='train_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_joint_motion_file',
        default='train_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_bone_file',
        default='train_rotation_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_bone_motion_file',
        default='train_rotation_motion_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--train_label_file',
        default='train_label.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_joint_file',
        default='test_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_joint_motion_file',
        default='test_joint_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_bone_file',
        default='test_rotation_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_bone_motion_file',
        default='test_rotation_motion_data.npy',
        type=str,
        metavar='PATH'
    )
    parser.add_argument(
        '--test_label_file',
        default='test_label.npy',
        type=str,
        metavar='PATH'
    )

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.train_joint_file = os.path.join(args.data_rootdir, args.train_joint_file)
    args.train_bone_file = os.path.join(args.data_rootdir, args.train_bone_file)
    args.train_joint_motion_file = os.path.join(args.data_rootdir, args.train_joint_motion_file)
    args.train_bone_motion_file = os.path.join(args.data_rootdir, args.train_bone_motion_file)
    args.train_label_file = os.path.join(args.data_rootdir, args.train_label_file)
    args.test_joint_file = os.path.join(args.data_rootdir, args.test_joint_file)
    args.test_bone_file = os.path.join(args.data_rootdir, args.test_bone_file)
    args.test_joint_motion_file = os.path.join(args.data_rootdir, args.test_joint_motion_file)
    args.test_bone_motion_file = os.path.join(args.data_rootdir, args.test_bone_motion_file)
    args.test_label_file = os.path.join(args.data_rootdir, args.test_label_file)

    return args


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    train_TwoStreamDGNN(args)