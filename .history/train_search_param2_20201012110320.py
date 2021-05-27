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

#from resnet_change2 import *
from resnet_change2 import *
from prune_params2 import ResNet20_Channel_Prune
from net_measure import measure_model, measure_param
from utils import AverageMeter, accuracy, count_parameters_in_MB, save_checkpoint
from architect1 import Architect


parser = argparse.ArgumentParser(description='Cifar10 Train Search')
parser.add_argument('--data', type=str, default='/home/cyh/workspace/cifar10',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001,
                    help='min learning rate(0.0)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
#parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
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
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
args = parser.parse_args()

log = open(os.path.join(args.save, 'channel-search-resnet20-0915.txt'),'w')

prune_index = ResNet20_Channel_Prune.index
prune_ratio = ResNet20_Channel_Prune.prune_ratio
#max2_ratio = torch.zeros(len(prune_index), 3)
min_ratio = torch.zeros(len(prune_index), 3)
min_ratio[:, 2] = -1
channel16 = list(range(2, 17, 2))
channel32 = list(range(2, 33, 2))
channel64 = list(range(2, 65, 2))

def main():
    if not torch.cuda.is_available():
        print('no gpu device available!!!')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    print_log('=> parameters: {}'.format(args), log)

    best_acc = 0
    best_epoch = 0
    base_epoch = 0
    criterion = nn.CrossEntropyLoss().cuda()

    count_ops, count_params, conv_list, other_list = measure_model(depth=20)
    print('=> count_ops: {}, count_params: {}, conv list: {}, other list: {}'.format(count_ops, count_params, conv_list, other_list))

    model = resnet(depth=20, cfg=[]).cuda()
    model_dict = model.state_dict()
    #model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    '''arch_optimizer = torch.optim.Adam(model.module.arch_parameters(), lr=args.arch_learning_rate,
                                    betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)'''

    train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                        ])
    valid_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    train_data = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

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

    architect = Architect(model, criterion, count_params, conv_list, other_list, args)

    print_log('==> arch parameters: {}'.format(model.arch_parameters()), log)
    print_log('==> arch parameters ratio: {}'.format(F.softmax(model.arch_params, dim=-1)), log)

    for epoch in range(args.epochs):
        #scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print_log('=> epoch {}, lr {}'.format(epoch, lr), log)

        if args.change and epoch - base_epoch >= 15:
            arch_weights = F.softmax(model.arch_params, dim=-1)
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

            if epoch % 35 == 0  and epoch != 0:
                base_epoch = 35 * (epoch // 35)
                for j in range(len(prune_index)):
                    max_ratio = prune_ratio[j][index[j][0].item()]
                    a = prune_ratio[j][index[j][3].item()]
                    num = 0
                    if j < 7:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-3, 0), min(max_ratio+3, len(channel16)-1))
                            num += 1
                    elif j < 13:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-5, 0), min(max_ratio+5, len(channel32)-1))
                            num += 1
                    else:
                        while(a in prune_ratio[j] and num <= 3):
                            a = random.randint(max(max_ratio-11, 0), min(max_ratio+11, len(channel64)-1))
                            num += 1
                    if min_ratio[j][0] >= 10:
                        prune_ratio[j][index[j][3].item()] = a
                        min_ratio[j][0] = 0
                        min_ratio[j][2] = a
                    else:
                        min_ratio[j][0] = 0
                    ratios = 1e-3 * torch.randn(1, 4)
                    with torch.no_grad():
                        for k in range(4):
                            model.arch_params[j][k] = ratios[0][k].item()
                with torch.no_grad():
                    model.load_state_dict(model_dict)

        train_acc, train_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, base_epoch, count_params, count_ops, conv_list)
        scheduler.step()
        if epoch == 0:
            params = {}
            for k, v in model.named_parameters():
                params[k] = v
            model_dict.update(params)

        print_log('=> train acc: {}'.format(train_acc), log)
        print_log('=> min ratio: {}'.format(min_ratio), log)
        print_log('=> arch parameters ratio: {}'.format(F.softmax(model.arch_params, dim=-1)),log)
        print_log('=> prune ratio: {}'.format(prune_ratio), log)

        if args.epochs - epoch <= 1:
            valid_acc, valid_loss = infer(valid_queue, model, criterion)
            print_log('valid_acc: {}'.format(valid_acc), log)

    arch_weights = F.softmax(model.arch_params, dim=-1)
    _, index = arch_weights.topk(1, 1, True, True)
    max_cfg = []
    #mix_cfg = []
    for j in range(len(prune_index)):
        if j < 7:
            channel = channel16[prune_ratio[j][index[j][0].item()]]
            max_cfg.append(channel)
        elif j < 13:
            channel = channel32[prune_ratio[j][index[j][0].item()]]
            max_cfg.append(channel)
        else:
            channel = channel64[prune_ratio[j][index[j][0].item()]]
            max_cfg.append(channel)
      
        '''channel = max(int(round(cfg[j] * (1 - prune_ratio[j][index[j][0].item()]) / 2) * 2), 2)
        max_cfg.append(channel)
        mix_prune_ratio = 0
        for k in range(5):
            mix_prune_ratio += prune_ratio[j][k] * arch_weights[j][k].item()
            #mix_channel += max(int(round(cfg[j] * (1 - prune_ratio[j][k]) * arch_weights[j][k].item() / 2) * 2), 2)
        mix_channel = max(int(round(cfg[j] * (1 - mix_prune_ratio) / 2) * 2), 2)
        mix_cfg.append(mix_channel)'''
    print_log('=> original max cfg: {}'.format(max_cfg), log)
    for j in range(len(max_cfg)):
        if j % 2 == 0 and j != 0:
            if max_cfg[j] < max_cfg[j-2]:
                max_cfg[j] = max_cfg[j-2]
    print_log('==> max cfg: {}'.format(max_cfg), log)
    #print_log('==> mix cfg: {}'.format(mix_cfg), log)
    print_log("==> arch parameters： {}".format(model.arch_parameters()), log)
    #print_log('==> best acc: {}, best epoch: {}'.format(best_acc, best_epoch), log)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, base_epoch, count_params, count_ops, conv_list):
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
        #input_search, target_search = next(iter(valid_queue))
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

        if epoch - base_epoch >= 15:
            arch_loss, arch_basic_loss, arch_param_loss = architect.step(inputs, targets, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            arch_losses.update(arch_loss.item(), input_search.size(0))
            arch_basic_losses.update(arch_basic_loss.item(), input_search.size(0))
            arch_param_losses.update(arch_param_loss, input_search.size(0))


        if index % args.report_freq == 0:
            print_log('=> time: {}, train index: {}, loss: {}, top1: {}, top5: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), index, losses.avg, top1.avg, top5.avg), log)

        if index % args.report_freq == 0 and epoch - base_epoch >= 15:
            print_log('=> time: {}, train index: {}, arch loss: {}, basic loss: {}, param loss: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), index, arch_losses.avg, arch_basic_losses.avg, arch_param_losses.avg), log)


    return top1.avg, losses.avg

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

def count_model_params(model):
    arch_weights = F.softmax(model.module.arch_params, dim=-1)
    _, index = arch_weights.topk(1, 1, True, True)
    cfg = []
    for k, m in enumerate(model.module.modules()):
        if k in prune_index:
            index_p = prune_index.index(k)
            if index_p < 7:
                channel = channel16[prune_ratio[index_p][index[index_p][0].item()]]
                cfg.append(channel)
            elif index_p < 13:
                channel = channel32[prune_ratio[index_p][index[index_p][0].item()]]
                cfg.append(channel)
            else:
                channel = channel64[prune_ratio[index_p][index[index_p][0].item()]]
                cfg.append(channel)

            '''pr = prune_ratio[index_p][index[index_p][0].item()]
            oC = max(int(round((m.weight.data.shape[0] * (1 - pr)) / 2) * 2), 2)
            cfg.append(oC)'''
    total = measure_param(depth=20, cfg=cfg)
    return total

    '''total = sum(p.numel() for p in model.parameters())
    arch_weights = F.softmax(model.module.arch_params, dim=-1)
    _, index = arch_weights.topk(1, 1, True, True)
    for k, m in enumerate(model.module.modules()):
        if k in prune_index:
            index_p = prune_index.index(k)
            if index_p == 0 :
                pr = prune_ratio[index_p][index[index_p][0].item()]
                oC = m.weight.data.shape[0] - int(round((m.weight.data.shape[0] * (1 - pr)) / 2) * 2)
                total -= oC * m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3]
                #total -= int(m.weight.data.numel() * (1 - prune_ratio[index_p][index[index_p][0].item()]))
            else:
                pr0 = prune_ratio[index_p-1][index[index_p-1][0].item()]
                pr1 = prune_ratio[index_p][index[index_p][0].item()]
                iC = m.weight.data.shape[1] - int(round((m.weight.data.shape[1] * (1 - pr0)) / 2) * 2)
                oC = m.weight.data.shape[0] - int(round((m.weight.data.shape[0] * (1 - pr1)) / 2) * 2)
                total -= oC * iC * m.weight.data.shape[2] * m.weight.data.shape[3]

            #total -= int(m.weight.data.numel() * (1 - prune_ratio[index_p][index[index_p][0].item()]))
    return total'''

def count_model_flops(model, total_flops, conv_list):
    arch_weights = F.softmax(model.module.arch_params, dim=-1)
    _, index = arch_weights.topk(1, 1, True, True)
    total = total_flops
    #print(total)
    #print('=> prune index: {}'.format(prune_index))
    for k, m in enumerate(model.module.modules()):
        if k in prune_index:
            if k == 1:
                pr = prune_ratio[0][index[0][0].item()]
                total -= int(conv_list[0] // 2 * pr)
                #print('=> total: {}'.format(total))
            elif k == 6:
                pr0 = 1 - prune_ratio[0][index[0][0].item()]
                pr1 = 1 - prune_ratio[1][index[1][0].item()]
                total -= int(conv_list[1] // 2 * (1 - pr0 * pr1))
                #print('=> total: {}'.format(total))
            else:
                index_p = prune_index.index(k)
                pr = prune_ratio[index_p][index[index_p][0].item()]
                total -= int(conv_list[2*index_p-1] // 2 * pr)
                #print('=> total: {}'.format(total))
        elif k-3 in prune_index and k-3 != 1:
                index_p = prune_index.index(k-3)
                pr = prune_ratio[index_p][index[index_p][0].item()]
                total -= int(conv_list[2*index_p] // 2 * pr)
                #print('=> total: {}'.format(total))
           
           
    return total

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == '__main__':
    main()
