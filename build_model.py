from cfg_parse import parse_cfg
import torch.nn as nn

model_list = parse_cfg("D:\ObjDet\cfgs\darknet53_DeformConv.cfg")

def create_blocks(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filter = 3
    output_filter = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if(x["type"] == "convolutional"):
            activation = x['activation']
            try:
                batch_norm = int(x['batch_norm'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x['padding'])
            kernel = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filter,filters,kernel,stride,pad,bias=bias)
            module.add_module(f"conv_{i}",conv)

            


