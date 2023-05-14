#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
	#self.sa = SpatialAttention()

        self.self_attn1 = Self_Attn( 32, 'relu')
        self.self_attn2 = Self_Attn( 32, 'relu')
        self.self_attn3 = Self_Attn( 32, 'relu')    
        self.self_attn4 = Self_Attn( 32, 'relu')    
        self.self_attn5 = Self_Attn( 32, 'relu')                              
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
#        self.res_conv1 = nn.Sequential(
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU()
#            )
#        self.res_conv2 = nn.Sequential(
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU()
#            )
#        self.res_conv3 = nn.Sequential(
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU()
#            )
#        self.res_conv4 = nn.Sequential(
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU()
#            )
#        self.res_conv5 = nn.Sequential(
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.ReLU()
#            )
#        self.conv_i = nn.Sequential(
#            nn.Conv2d(32 + 32, 32, 3, 1, 1),
#            nn.Sigmoid()
#            )
#        self.conv_f = nn.Sequential(
#            nn.Conv2d(32 + 32, 32, 3, 1, 1),
#            nn.Sigmoid()
#            )
#        self.conv_g = nn.Sequential(
#            nn.Conv2d(32 + 32, 32, 3, 1, 1),
#            nn.Tanh()
#            )
#        self.conv_o = nn.Sequential(
#            nn.Conv2d(32 + 32, 32, 3, 1, 1),
#            nn.Sigmoid()
#            )
        self.res_conv1E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )
        self.res_conv1_1E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )            
        self.res_conv1D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )
        self.res_conv1_1D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )      
            
        self.res_conv2E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )
        self.res_conv2_1E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )            
        self.res_conv2D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )
        self.res_conv2_1D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )

        self.res_conv3E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )
        self.res_conv3_1E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )            
        self.res_conv3D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )
        self.res_conv3_1D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )
                                          
 
        self.res_conv4E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )
        self.res_conv4_1E = nn.Sequential(
			nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),
            )            
        self.res_conv4D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )
        self.res_conv4_1D = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
			nn.ReLU(inplace=True),						
            )

                               
        self.conv_i = nn.Sequential(
			# dw
			nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
			# dw
			nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
			# dw
			nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
			# dw
			nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            
            resx = x
            e1 = self.res_conv1E(x) 
            e1_1 = self.res_conv1_1E(e1)                       
            e1_1_a = self.self_attn1(e1_1)
            d1 = self.res_conv1D(e1_1_a+e1_1) 
            d1_1 = self.res_conv1_1D(d1+e1)              
            x = F.relu(d1_1 + resx)

            resx = x
            e2 = self.res_conv2E(x) 
            e2_1 = self.res_conv2_1E(e2)                       
            e2_1_a = self.self_attn2(e2_1)
            d2 = self.res_conv2D(e2_1_a+e2_1) 
            d2_1 = self.res_conv2_1D(d2+e2)              
            x = F.relu(d2_1 + resx)

            resx = x
            e3 = self.res_conv3E(x) 
            e3_1 = self.res_conv3_1E(e3)                       
            e3_1_a = self.self_attn3(e3_1)
            d3 = self.res_conv3D(e3_1_a+e3_1) 
            d3_1 = self.res_conv3_1D(d3+e3)              
            x = F.relu(d3_1 + resx)        

            resx = x
            e4 = self.res_conv4E(x) 
            e4_1 = self.res_conv4_1E(e4)                       
            e4_1_a = self.self_attn4(e4_1)
            d4 = self.res_conv4D(e4_1_a+e4_1) 
            d4_1 = self.res_conv4_1D(d4+e4)              
            x = F.relu(d4_1 + resx)                  
 

            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PReNet_LSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x1 = x
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x_list.append(x)

        return x, x_list


class PReNet_GRU(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_GRU, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_z = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        # self.conv_o = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        #     )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x1 = torch.cat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = torch.cat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_x(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_x, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            #x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list


## PRN
class PRN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PRN_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PRN_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list
