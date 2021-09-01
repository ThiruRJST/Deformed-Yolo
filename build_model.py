from cfg_parse import parse_cfg
from custom_ops import DeformableConv2d, EmptyLayer
import torch.nn as nn
from torchsummary import summary
model_list = parse_cfg("cfgs/darknet53_DeformConv.cfg")
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
                batch_norm = int(x['batch_norm'])
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

        elif(x['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("Shortcut_{}".format(i), shortcut)

        module_list.append(module)
        prev_filter = filters
        output_filter.append(filters)
    return module_list


# print(create_blocks(model_list))
