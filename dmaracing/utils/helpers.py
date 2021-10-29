import yaml

class CmdLineArguments:
    pass

def getcfg(path):
    pth_cfg = path+'/cfg.yml'
    pth_cfg_train = path+'/cfg_train.yml'
    
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    logdir = 'logs/'+cfg_train['learn']['experiment']
    return cfg, cfg_train, logdir
