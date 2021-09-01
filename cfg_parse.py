def parse_cfg(cfgfile):
    """
    Takes a config file ,parses it line by line
    It creates blocks from the config file and appends it to list of modules 
    creating the layers for the network.
    
    """

    block = {}
    blocks = []

    files = open(cfgfile,'r')
    lines = files.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] !="#"]
    lines = [x.rstrip().lstrip() for x in lines]


    for line in lines:
        if line[0] == "[":
            if len(block) !=0:
                #print(block)
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
        
    blocks.append(block)

    return blocks




