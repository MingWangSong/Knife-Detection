# -*- coding: utf-8 -*-
# @Time    : 2020-09-16 13:23
# @Author  : Wily
# @File    : param.py
# @Software: PyCharm Community Edition

import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
# from models.build_model import build_model
from networks_mDcn import *
import numpy as np

# model = build_model(x='pyconvresnet', y=50)

#model = PReNet(6,False)
#print(model)
#resnet = model

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.4fM' % (total / 1e6))

def print_model_parm_flops(input_v,model):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per
    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)
    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
        2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)
    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
    foo(model)
    #input = Variable(torch.rand(3, 100, 100).unsqueeze(0), requires_grad=True)
    out = model(input_v)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('Number of FLOPs: %.4fG' % (total_flops / 1e9))

#print_model_parm_nums()
#print_model_parm_flops()
