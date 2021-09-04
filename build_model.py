from cfg_parse import parse_cfg
from custom_ops import DeformableConv2d, EmptyLayer
import torch.nn as nn
from torchsummary import summary

# print(model_list)


def create_blocks(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filter = 3
    output_filter = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if(x["type"] == "convolutional"):
            # print("0")
            activation = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x['pad'])
            kernel = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filter, filters, kernel,
                             stride, pad, bias=bias)
            module.add_module("conv_{}".format(i), conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(i), bn)

            if activation == "mish":
                activation_fn = nn.Mish()
                module.add_module("Mish_{}".format(i), activation_fn)
            elif activation == "leaky":
                activation_fn = nn.LeakyReLU()
                module.add_module("Leaky_ReLU{}".format(i),activation_fn)

        elif(x['type'] == "deformable"):
            # print("1")

            activation = x['activation']
            try:
                batch_norm = int(x['batch_norm'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x['pad'])
            kernel = int(x['size'])
            stride = int(x['stride'])

            dcn = DeformableConv2d(
                in_channels=prev_filter,
                out_channels=filters,
                kernel=kernel,
                stride=stride, padding=padding, bias=bias
            )

            module.add_module("dcn_{}".format(i), dcn)
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(i), bn)

            if activation == "mish":
                activation_fn = nn.Mish()
                module.add_module("Mish_{}".format(i), activation_fn)
            elif activation == "leaky":
                activation_fn = nn.LeakyReLU()
                module.add_module("Leaky_ReLU{}".format(i),activation_fn)

        elif(x['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("Shortcut_{}".format(i), shortcut)
        
        elif(x['type'] == 'avgpool'):
            pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            module.add_module("AvgPool{}".format(i),pool)

        module_list.append(module)
        prev_filter = filters
        output_filter.append(filters)
    return module_list


class Deformed_Darknet53(nn.Module):

    def __init__(self):
        super(Deformed_Darknet53, self).__init__()

        self.model_list = parse_cfg("cfgs/new-darknet.cfg")
        self.module_list = create_blocks(self.model_list)
        #print(self.module_list)

    def forward(self, x):
        outputs = {}
        for i, module in enumerate(self.model_list[1:]):
            module_type = module['type']
            if module_type == "convolutional" or module_type == "deformable" or module_type == "avgpool":
                x = self.module_list[i](x)

            elif module_type == "shortcut":
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            

            outputs[i] = x

        return x


