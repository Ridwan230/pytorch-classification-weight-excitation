import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class Conv2d_Custom(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_Custom, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)

        self.count = 0
        planes = in_channels
        self.fc1 = nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1)
        self.fc2 = nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
        self.eps = 1e-5

    def forward(self, x):
        # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight

        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                           keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_avg
        std = weight.view(weight.size(
            0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)

        # LWE
        wght = F.avg_pool2d(weight, weight.size(2))
        wght = F.relu(self.fc1(wght))
        wght = F.sigmoid(self.fc2(wght))
        weight = weight * wght

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# class Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)
#         self.count = 0
#         self.eps = 1e-5

#     def forward(self, x):
#         # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
#         weight = self.weight

#         # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
#         weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                                             keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_mean
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)
#         # MWE
#         max_val = (1.0 + 0.1) * \
#             torch.max(torch.max(weight), -1 * torch.min(weight))
#         if self.count:
#             weight = max_val * 0.5 * \
#                 torch.log((1+weight/max_val) / (1-weight/max_val))
#         else:
#             self.count = 1
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# CBAM
# class Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)

#         self.count = 0
#         planes = in_channels
#         self.eps = 1e-5
#         ratio=16

#         #modify
#         self.mlp = nn.Sequential(
#             nn.Conv2d(planes//groups, max(planes//16,1), kernel_size=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(max(planes//16,1), planes//groups, kernel_size=1)

#             # nn.Linear(in_channels, max(in_channels//ratio,1), bias=False),
#             # nn.Linear(max(in_channels//ratio,1), in_channels, bias=False)
#         )

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
#         self.sigmoid=nn.Sigmoid()

#     def forward(self, x):
#         # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
#         weight = self.weight

#         # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
#         weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)


#         wght1 = self.avg_pool(weight)
#         #size(2) means kernel size as weight().def __init__(self, in_channels->0, out_channels, kernel_size->2, stride=1, padding=0, dilation=1, groups=1, bias=True):

#         wght1 = self.mlp(wght1) #cX1X1

#         wght2 = self.max_pool(weight)

#         wght2 = self.mlp(wght2) #c*1*1

#         wght = torch.sigmoid(wght1+wght2)

#         skip_1 = weight * wght

#         weight = weight + skip_1


#         #spatial attention
#         max_result,_=torch.max(weight,dim=1,keepdim=True)
#         avg_result=torch.mean(weight,dim=1,keepdim=True)
#         result=torch.cat([max_result,avg_result],1)
#         wght_2=self.conv(result)
#         wght_2=self.sigmoid(wght_2)

#         skip_2 = weight*wght_2

#         weight=weight + skip_1 + skip_2
# #         wght3 = self.avg_pool(weight)
# #         wght4 = self.max_pool(weight)
# #         wght5 = torch.cat([wght3,wght4],1)
# #         wght_2 = torch.sigmoid(result)
# #         weight = weight * wght_2


#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# Implementation for Conv2D with LWE included.
class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

        self.count = 0
        planes = in_channels
        self.fc1 = Conv2d_Custom(
            planes//groups, max(planes//16, 1), kernel_size=1)
        self.fc2 = Conv2d_Custom(
            max(planes//16, 1), planes//groups, kernel_size=1)
        self.eps = 1e-5

    def forward(self, x):
        # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight

        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                           keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_avg
        std = weight.view(weight.size(
            0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)

        # LWE
        wght = F.avg_pool2d(weight, weight.size(2))
        wght = F.relu(self.fc1(wght))
        wght = F.sigmoid(self.fc2(wght))
        weight = weight * wght

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Implementation for Conv3D with LWE included.


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

        self.count = 0
        planes = in_channels
        self.fc1 = nn.Conv3d(planes, max(planes//16, 1), kernel_size=1)
        self.fc2 = nn.Conv3d(max(planes//16, 1), planes, kernel_size=1)
        self.eps = 1e-5

    def forward(self, x):
        # return super(Conv3d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight

        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                           keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_avg
        std = weight.view(weight.size(
            0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)

        # LWE
        wght = F.avg_pool3d(weight, weight.size(2))
        wght = F.relu(self.fc1(wght))
        wght = F.sigmoid(self.fc2(wght))
        weight = weight * wght

        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Other implementations
'''
# Implementation for Conv2D with MWE included. 
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.count = 0
        self.eps = 1e-5
    def forward(self, x):
        # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight
        
        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        # MWE
        max_val = (1.0 + 0.1) * torch.max(torch.max(weight), -1* torch.min(weight)) 
        if self.count: 
            weight = max_val * 0.5 * torch.log ((1+weight/max_val) / (1-weight/max_val))
        else: 
            self.count =1
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# Implementation for Conv2D with LWE and MWE included. 
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.count = 0
        planes = in_channels
        self.fc1 = nn.Conv2d(planes//groups, max(planes//16,1), kernel_size=1)
        self.fc2 = nn.Conv2d(max(planes//16,1), planes//groups, kernel_size=1)
        self.eps = 1e-5
    def forward(self, x):
        # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight
        
        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        
        # LWE 
        w = F.avg_pool2d(weight, weight.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        weight = weight * w
        # MWE
        max_val = (1.0 + 0.1) * torch.max(torch.max(weight), -1* torch.min(weight)) 
        if self.count: 
            weight = max_val * 0.5 * torch.log ((1+weight/max_val) / (1-weight/max_val))
        else: 
            self.count =1
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# Implementation for Conv3D with LWE and MWE included. 
class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.count = 0
        planes = in_channels
        self.fc1 = nn.Conv3d(planes, max(planes//16,1), kernel_size=1)
        self.fc2 = nn.Conv3d(max(planes//16,1), planes, kernel_size=1)
        self.eps = 1e-5
    def forward(self, x):
        # return super(Conv3d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight
        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        
        # LWE 
        w = F.avg_pool3d(weight, weight.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        weight = weight * w
        # MWE
        max_val = (1.0 + 0.1) * torch.max(torch.max(weight), -1* torch.min(weight)) 
        if self.count: 
            weight = max_val * 0.5 * torch.log ((1+weight/max_val) / (1-weight/max_val))
        else: 
            self.count =1
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
'''
