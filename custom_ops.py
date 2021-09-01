import torch
from torchvision.ops import deform_conv2d
import torch.nn as nn


class DeformableConv2d(nn.Module):
    def __init__(self,
    in_channels,
    out_channels,
    kernel = 3,
    stride = 1,
    padding = 1,
    bias = False
    ) :
        super(DeformableConv2d,self).__init__()
        
        if type(stride) == tuple:
            self.stride = stride
        else:
            self.stride = stride

        self.padding = padding

        #Offset Convolution
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel * kernel,
            kernel_size = kernel,
            stride = stride,
            padding = self.padding,
            bias = True
        )

        nn.init.constant_(self.offset_conv.weight,0.0)
        nn.init.constant_(self.offset_conv.bias,0.0)

        #Modulator Convolution
        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel * kernel,
            kernel_size = kernel,
            stride = stride,
            padding = self.padding,
            bias = True
            )

        nn.init.constant_(self.modulator_conv.weight,0.0)
        nn.init.constant_(self.modulator_conv.bias,0.0)

        #Regular Convolutions
        self.regular_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel,
            stride = stride,
            padding = self.padding,
            bias = bias
        )

    def forward(self,x):
        
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        x = deform_conv2d(input=x,
        offset = offset,
        weight = self.regular_conv.weight,
        bias = self.regular_conv.bias,
        padding = self.padding,
        mask = modulator,
        stride = self.stride
        )

        return x
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
