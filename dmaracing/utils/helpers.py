import yaml
import os
import torch

class CmdLineArguments:
    pass


def set_dependent_cfg_entries(cfg):
    numObservations = cfg['sim']['numConstantObservations']
    numObservations += cfg['learn']['horizon']*2 #lookaheadhorizon
    numObservations += (2+1+1+2) * (cfg['sim']['numAgents']-1)
    cfg['sim']['numObservations'] = numObservations

def getcfg(path):
    pth_cfg = path+'/cfg.yml'
    pth_cfg_train = path+'/cfg_train.yml'
    
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    logdir = 'logs/'+cfg_train['runner']['experiment_name']
    
    #update parameter dependent observation counts
    set_dependent_cfg_entries(cfg)
    return cfg, cfg_train, logdir


def  get_run(logdir, run, chkpt):
    
    if type(run) == str:
        model_dir = logdir + '/'+ run
    else:
        runs = os.listdir(logdir)
        runs.sort()
        runstr = runs[run]
        model_dir = logdir+'/'+ runstr

    if chkpt == -1:
        models = os.listdir(model_dir)
        models = [i for i in models if 'model' in i]
        nrs = [int(model[6:-3]) for model in models]
        nrs.sort()
        modelnr = int(nrs[-1])
    else:
        modelnr = chkpt

    return model_dir, modelnr

def rand(min, max, shape, device):
    r = torch.rand(shape, device=device, dtype = torch.float, requires_grad=False)
    dist = max-min
    return dist*r + min