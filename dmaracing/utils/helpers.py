from numbers import Rational
import yaml
import os
import torch

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)

class CmdLineArguments:
    def __init__(self):
        self.device = 'cpu'
        self.headless = False
        self.override_cfg_train = False
        self.override_keys = []
        self.override_values = []

    def parse(self, argv):
        for arg in argv:
            split = arg.split('=')
            self.override_keys.append(split[0])
            self.override_values.append(split[1])
            
    def override_cfg_with_args(self, cfg, cfg_train):
        overridestring_train = ''
        overridestring_env = ''
        for override_key, override_value in zip(self.override_keys, self.override_values):
            print(override_key)
            if override_key == 'headless':
                self.headless = False if override_value == 'False' else True
            for key, val in cfg_train.items():
                if key == override_key:
                    #typecast from string into type in dict
                    cfg_train[key] = type(val)(override_value)
                    print('cfg_train: '+ key + ' changed to', override_value)
                    overridestring_train+=key+override_value
                elif isinstance(cfg_train[key], dict):
                    for key2, val2 in cfg_train[key].items():
                        #print(key2, override_key)
                        if key2 == override_key:
                            #typecast from string into type in dict
                            cfg_train[key][key2] = type(val2)(override_value)
                            print('cfg_train: '+ key+':' + key2 + ' changed to', override_value)
                            overridestring_train+=key+key2+override_value

        for override_key, override_value in zip(self.override_keys, self.override_values):
            print(override_key)
            for key, val in cfg.items():
                if key == override_key:
                    #typecast from string into type in dict
                    cfg[key] = type(val)(override_value)
                    print('cfg: '+ key + ' changed to', override_value)
                    overridestring_train+=key+override_value
                elif isinstance(cfg[key], dict):
                    for key2, val2 in cfg[key].items():
                        #print(key2, override_key)
                        if key2 == override_key:
                            #typecast from string into type in dict
                            cfg[key][key2] = type(val2)(override_value)
                            print('cfg: '+ key+':' + key2 + ' changed to', override_value)
                            overridestring_env+=key+key2+override_value
        if len(overridestring_train):
            cfg_train['overrides'] = overridestring_train

        if len(overridestring_env):
            cfg['overrides'] = overridestring_env


def set_dependent_cfg_entries(cfg, cfg_train):
    numObservations = cfg['sim']['numConstantObservations']
    # numObservations += cfg['learn']['horizon']*2 #lookaheadhorizon
    numObservations += cfg['learn']['horizon']*4 #lookaheadhorizon
    numObservations += (2+1+2+2) * (cfg['sim']['numAgents']-1)
    cfg['sim']['numObservations'] = numObservations
    if cfg_train['policy']['attentive']:
        # cfg_train['policy']['num_ego_obs'] = cfg['learn']['horizon']*2 + cfg['sim']['numConstantObservations']
        cfg_train['policy']['num_ego_obs'] = cfg['learn']['horizon']*4 + cfg['sim']['numConstantObservations']
        cfg_train['policy']['num_ado_obs'] = 2+1+2+2
    

def getcfg(path, postfix = ''):
    pth_cfg = path+'/cfg.yml'
    pth_cfg_train = path+'/cfg_train' + postfix + '.yml'
    
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    logdir = 'logs/'+cfg_train['runner']['experiment_name']
    
    #update parameter dependent observation counts
    set_dependent_cfg_entries(cfg, cfg_train)
    return cfg, cfg_train, logdir


def  get_run(logdir, run, chkpt):
    
    if type(run) == str:
        model_dir = logdir + '/'+ run
    else:
        runs = os.listdir(logdir)
        runs.sort()
        runstr = runs[run]
        model_dir = logdir+'/'+ runstr

    if chkpt < 0:
        models = os.listdir(model_dir)
        models = [i for i in models if 'model' in i]
        nrs = [int(model[6:-3]) for model in models]
        nrs.sort()
        modelnr = int(nrs[chkpt])
    else:
        modelnr = chkpt

    return model_dir, modelnr

def  get_all_chckpts_run(logdir, run):
    
    if type(run) == str:
        model_dir = logdir + '/'+ run
    else:
        runs = os.listdir(logdir)
        runs.sort()
        runstr = runs[run]
        model_dir = logdir+'/'+ runstr

    models = os.listdir(model_dir)
    models = [i for i in models if 'model' in i]
    nrs = [int(model[6:-3]) for model in models]
    nrs.sort()
    
    return model_dir, nrs

def rand(min, max, shape, device):
    r = torch.rand(shape, device=device, dtype = torch.float, requires_grad=False)
    dist = max-min
    return dist*r + min

