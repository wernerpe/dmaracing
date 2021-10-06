import yaml

def getcfg(path):
    pth_cfg = path+'cfg.yml'
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg, []