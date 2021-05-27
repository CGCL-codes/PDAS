import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_measure import measure_param
#from prune_params1 import ResNet20_Channel_Prune
from prune_params1 import ResNet32_Channel_Prune
from utils import AverageMeter

'''prune_index = ResNet20_Channel_Prune.index
prune_ratio = ResNet20_Channel_Prune.prune_ratio'''

prune_index = ResNet32_Channel_Prune.index
prune_ratio = ResNet32_Channel_Prune.prune_ratio
channel16 = list(range(2, 17, 2))
channel32 = list(range(2, 33, 2))
channel64 = list(range(2, 65, 2))
default_cfg = [16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):

    def __init__(self, model, criterion, total_ops, conv_list, other_list, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion
        self.total_ops = total_ops
        self.conv_list = conv_list
        self.other_list = other_list
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, inputs, targets, eta, network_optimizer):
        logits = self.model(inputs)
        basic_loss = self.criterion(logits, targets)

        arch_weights = F.softmax(self.model.arch_params, dim=-1)
        _, index = arch_weights.topk(1, 1, True, True)
        ratio = [(prune_ratio[j][index[j][0].item()] + 1) * 2 / default_cfg[j] for j in range(len(prune_index))]
        count_params = 0
        for j in range(len(ratio)):
            if j == 0:
                count_params += self.conv_list[j] * ratio[j]
            else:
                count_params += self.conv_list[2*j-1] * ratio[j-1] * ratio[j] + self.conv_list[2*j] * ratio[j]
        count_params += sum(self.other_list)
        if count_params > (0.63 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        elif count_params < (0.57 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        else:
            param_loss = 0

        loss = 0.4*basic_loss + 0.6*param_loss
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            loss, basic_loss, param_loss = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            loss, basic_loss, param_loss = self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        return loss, basic_loss, param_loss

    def _backward_step(self, input_valid, target_valid):

        logit_valid = self.model(input_valid)
        basic_loss = self.criterion(logit_valid, target_valid)

        arch_weights = F.softmax(self.model.arch_params, dim=-1)
        _, index = arch_weights.topk(1, 1, True, True)
        ratio = [(prune_ratio[j][index[j][0].item()] + 1) * 2 / default_cfg[j] for j in range(len(prune_index))]
        count_ops = 0
        for j in range(len(ratio)):
            if j == 0:
                count_ops += self.conv_list[j] * ratio[j]
            elif j == 1:
                count_ops += self.conv_list[2*j-1] * ratio[j-1] * ratio[j] + self.conv_list[2*j] * ratio[j]
            else:
                count_ops += (self.conv_list[2*j-1] + self.conv_list[2*j]) * ratio[j]
        count_ops += sum(self.other_list)
        if count_ops > (0.6925 * self.total_ops):
            param_loss = 3 * math.log(count_ops / (0.6625 * self.total_ops))
        elif count_ops < (0.6325 * self.total_ops):
            param_loss = -3 * math.log(count_ops / (0.6625 * self.total_ops))
        else:
            param_loss = 0

        loss = 0.5*basic_loss + 0.5*param_loss
        loss.backward()

        return loss, basic_loss, param_loss


    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        '''basic_losses = AverageMeter()
        param_losses = AverageMeter()
        losses = AverageMeter()'''

        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        logit_valid = unrolled_model(input_valid)
        basic_loss = self.criterion(logit_valid, target_valid)

        arch_weights = F.softmax(unrolled_model.arch_params, dim=-1)
        _, index = arch_weights.topk(1, 1, True, True)
        ratio = [(prune_ratio[j][index[j][0].item()] + 1) * 2 / default_cfg[j] for j in range(len(prune_index))]
        count_params = 0
        for j in range(len(ratio)):
            if j == 0:
                count_params += self.conv_list[j] * ratio[j]
            else:
                count_params += self.conv_list[2*j-1] * ratio[j-1] * ratio[j] + self.conv_list[2*j] * ratio[j]
        count_params += sum(self.other_list)
        if count_params > (0.63 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        elif count_params < (0.57 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        else:
            param_loss = 0

        unrolled_loss = 0.4*basic_loss + 0.6*param_loss
        #unrolled_loss = self.criterion(logit_valid, target_valid)

        unrolled_loss.backward()
        '''basic_losses.update(basic_loss.item(), input_valid.size(0))
        param_losses.update(param_loss, input_valid.size(0))
        losses.update(unrolled_loss.item(), input_valid.size(0)) '''

        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        #print('=> total loss: {}, basic loss: {}, param loss: {}'.format(losses.avg, basic_losses.avg, param_losses.avg)) 
        return unrolled_loss, basic_loss, param_loss

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, inputs, targets, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        logits = self.model(inputs)
        basic_loss = self.criterion(logits, targets)

        arch_weights = F.softmax(self.model.arch_params, dim=-1)
        _, index = arch_weights.topk(1, 1, True, True)
        ratio = [(prune_ratio[j][index[j][0].item()] + 1) * 2 / default_cfg[j] for j in range(len(prune_index))]
        count_params = 0
        for j in range(len(ratio)):
            if j == 0:
                count_params += self.conv_list[j] * ratio[j]
            else:
                count_params += self.conv_list[2*j-1] * ratio[j-1] * ratio[j] + self.conv_list[2*j] * ratio[j]
        count_params += sum(self.other_list)
        if count_params > (0.63 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        elif count_params < (0.57 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        else:
            param_loss = 0

        loss = 0.4*basic_loss + 0.6*param_loss
        #loss = self.criterion(logits, targets)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        logits = self.model(inputs)
        basic_loss = self.criterion(logits, targets)

        arch_weights = F.softmax(self.model.arch_params, dim=-1)
        _, index = arch_weights.topk(1, 1, True, True)
        ratio = [(prune_ratio[j][index[j][0].item()] + 1) * 2 / default_cfg[j] for j in range(len(prune_index))]
        count_params = 0
        for j in range(len(ratio)):
            if j == 0:
                count_params += self.conv_list[j] * ratio[j]
            else:
                count_params += self.conv_list[2*j-1] * ratio[j-1] * ratio[j] + self.conv_list[2*j] * ratio[j]
        count_params += sum(self.other_list)
        if count_params > (0.63 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        elif count_params < (0.57 * self.total_params):
            #param_loss = count_params / (0.6 * self.total_params)
            param_loss = 3 * math.log(count_params / (0.6 * self.total_params))
        else:
            param_loss = 0

        loss = 0.4*basic_loss + 0.6*param_loss
        #loss = self.criterion(logits, targets)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

def count_model_params(model):
    arch_weights = F.softmax(model.arch_params, dim=-1)
    _, index = arch_weights.topk(1, 1, True, True)
    cfg = []
    for k, m in enumerate(model.modules()):
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

    total = measure_param(depth=20, cfg=cfg)
    return total
