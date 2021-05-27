import os
import sys
import time
import glob
import math
import random
import logging
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from resnet164_change2 import *
#from prune_params2 import ResNet20_Channel_Prune
from prune_params2 import ResNet164_Channel_Prune
from net_measure import measure_model, measure_param
from utils import AverageMeter, accuracy, count_parameters_in_MB, save_checkpoint
from architect164_2 import Architect


parser = argparse.ArgumentParser(description='Cifar10 Train Search')
parser.add_argument('--data', type=str, default='/home/chenyuhao/workspace/cifar100',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.04,
                    help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0,
                    help='min learning rate(0.0)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
parser.add_argument('--save', type=str, default='./checkpoint/',
                    help='folder to save checkpoints and logs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5,
                    help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                    help='weight decay for arch encoding')
parser.add_argument('--change', action='store_true', default=False,
                    help='change prune ratio during searching')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--rounds', type=int, default=6, help='times to run process')
args = parser.parse_args()

log = open(os.path.join(args.save, 'channel-search-resnet110-0915.txt'),'w')

'''prune_index = ResNet20_Channel_Prune.index
prune_ratio = ResNet20_Channel_Prune.prune_ratio'''

prune_index = ResNet164_Channel_Prune.index
prune_ratio = ResNet164_Channel_Prune.prune_ratio

min_ratio = torch.zeros(len(prune_index), 3)
min_ratio[:, 2] = -1
channel16 = list(range(2, 17, 2))
channel32 = list(range(2, 33, 2))
channel64 = list(range(2, 65, 2))
cfg = [16, 8,8, 6,6, 16,16, 16,16, 16,16, 12,12, 12,12, 12,12, 4,4, 6,6, 12,12, 12,12, 6,6, 10,10, 16,16, 8,8, 8,8, 12,12, 14,14, 22,22, 14,14, 24,24, 14,14, 22,22, 32,32, 14,14, 22,22, 22,22, 22,22, 10,10, 14,14, 32,32, 32,32, 32,32, 10,10, 32,32, 64,64, 64,64, 44,44, 64,64, 64,64, 52,52, 64,64, 64,64, 24,24, 38,38, 8,8, 54,54, 26,26, 64,64, 64,64, 64,64, 24,24, 44,44]

default_cfg = []
default_cfg.append(cfg[0])
for j in range(3):
    for k in range(1, 19):
        default_cfg.append(cfg[36*j+2*k-1])
        default_cfg.append(cfg[36*j+2*k])
        if j == 0:
            default_cfg.append(16)
        elif j == 1:
            default_cfg.append(32)
        else:
            default_cfg.append(64)

base_model = resnet(depth=164, dataset='cifar100', cfg=cfg)
model_dict = base_model.state_dict()
final_cfg = []
final_param_loss = float('inf')

def main(round_num):
    if not torch.cuda.is_available():
        print('no gpu device available!!!')
        sys.exit(1)

    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    print_log('=> parameters: {}'.format(args), log)
    global final_param_loss, final_cfg
    best_acc = 0
    best_epoch = 0
    arch_param_loss = float('inf')
    train_param_loss = 0
    arch_cfg = []
    criterion = nn.CrossEntropyLoss().cuda()

    count_ops, count_params, conv_list, other_list = measure_model(depth=164, dataset='cifar100', cfg=default_cfg)
    print('=> count_ops: {}, count_params: {}, conv list: {}, other list: {}'.format(count_ops, count_params, conv_list, other_list))

    model = resnet(depth=164, dataset='cifar100', cfg=cfg)
    for k, m in enumerate(model.modules()):
        if(isinstance(m, nn.Conv2d)):
            print(k)
            print(m)
    model.load_state_dict(model_dict)
    model=torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                        ])
    valid_transform = transforms.Compose([
                            transforms.ToTensor(),
                            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                        ])
    train_data = datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, count_ops, conv_list, other_list, args)

    print_log('==> arch parameters: {}'.format(model.module.arch_parameters()), log)
    print_log('==> arch parameters ratio: {}'.format(F.softmax(model.module.arch_params, dim=-1)), log)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        print_log('=> epoch {}, lr {}'.format(epoch + (round_num * args.epochs), lr), log)

        train_acc, train_loss, train_param_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, count_params, count_ops, conv_list)
        scheduler.step()

        if epoch >= 3:
            arch_weights = F.softmax(model.module.arch_params, dim=-1)
            _, index = arch_weights.topk(1, 1, True, True)
            max_cfg = []
            for j in range(len(prune_index)):
                if j < 1:
                    channel = channel16[prune_ratio[j][index[j][0].item()]]
                    max_cfg.append(channel)
                elif j < 2:
                    channel = channel32[prune_ratio[j][index[j][0].item()]]
                    max_cfg.append(channel)
                else:
                    channel = channel64[prune_ratio[j][index[j][0].item()]]
                    max_cfg.append(channel)
            if train_acc >= best_acc or train_param_loss < arch_param_loss:
                arch_cfg = max_cfg
                best_acc = train_acc
                arch_param_loss = train_param_loss

        if args.change and epoch >= 3:
            arch_weights = F.softmax(model.module.arch_params, dim=-1)
            _, index = arch_weights.topk(4, 1, True, True)
            for j in range(len(prune_index)):
                new_index = prune_ratio[j][index[j][3].item()]
                old_index = min_ratio[j][1].item()
                count = min_ratio[j][0].item()
                if abs(new_index - old_index) < 1e-6:
                    min_ratio[j][0] += 1
                else:
                    min_ratio[j][0] = 0
                    min_ratio[j][1] = new_index

            if epoch == args.epochs-1:
                for j in range(len(prune_index)):
                    max_ratio = prune_ratio[j][index[j][0].item()]
                    a = prune_ratio[j][index[j][3].item()]
                    num = 0
                    if j < 1:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-1, 7), min(max_ratio+1, len(channel16)-1))
                            num += 1
                    elif j < 2:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-3, 7), min(max_ratio+3, len(channel32)-1))
                            num += 1
                    else:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-6, 15), min(max_ratio+11, len(channel64)-1))
                            num += 1
                    if min_ratio[j][0] >= 3:
                        prune_ratio[j][index[j][3].item()] = a
                        min_ratio[j][0] = 0
                        min_ratio[j][2] = a
                    else:
                        min_ratio[j][0] = 0
                    ratios = 1e-3 * torch.randn(1, 4)
                    with torch.no_grad():
                        for k in range(4):
                            model.module.arch_params[j][k] = ratios[0][k].item()

        print_log('=> train acc: {}'.format(train_acc), log)
        print_log('=> min ratio: {}'.format(min_ratio), log)
        print_log('=> arch parameters ratio: {}'.format(F.softmax(model.module.arch_params, dim=-1)),log)
        print_log('=> prune ratio: {}'.format(prune_ratio), log)

    if arch_param_loss <= final_param_loss:
        final_param_loss = arch_param_loss
        final_cfg = arch_cfg
    print_log('==> max cfg: {}'.format(arch_cfg), log)
    print_log("==> arch parameters： {}".format(model.module.arch_parameters()), log)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, count_params, count_ops, conv_list):
    losses = AverageMeter()
    basic_losses = AverageMeter()
    param_losses = AverageMeter()
    arch_losses = AverageMeter()
    arch_basic_losses = AverageMeter()
    arch_param_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    for index, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search, target_search = input_search.cuda(), target_search.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if epoch >= 3:
            arch_loss, arch_basic_loss, arch_param_loss = architect.step(inputs, targets, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            arch_losses.update(arch_loss.item(), input_search.size(0))
            arch_basic_losses.update(arch_basic_loss.item(), input_search.size(0))
            arch_param_losses.update(arch_param_loss, input_search.size(0))


        if index % args.report_freq == 0 or index + 1 >= len(train_queue):
            print_log('=> time: {}, train index: {}, loss: {}, top1: {}, top5: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), index, losses.avg, top1.avg, top5.avg), log)

        if (index % args.report_freq == 0 or index + 1 >= len(train_queue)) and epoch >= 3:
            print_log('=> time: {}, train index: {}, arch loss: {}, basic loss: {}, param loss: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), index, arch_losses.avg, arch_basic_losses.avg, arch_param_losses.avg), log)


    return top1.avg, losses.avg, arch_param_losses.avg

def infer(valid_queue, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        for index, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)
            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if index % args.report_freq == 0:
                print_log('=> time: {}, valid index: {}, loss: {}, top1: {}, top5: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), index, losses.avg, top1.avg, top5.avg), log)
    
    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == '__main__':
    for i in range(args.rounds):
        main(i)
        if final_param_loss < 0.1:
            break
    print('=> final cfg: {}'.format(final_cfg))
    print('=> final param loss: {}'.format(final_param_loss))
