import os
import torch
import numpy as np
import torchvision.transforms as transforms

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))





