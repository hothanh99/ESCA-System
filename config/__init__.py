from .default import get_cfg_defaults
import os, sys
sys.path.append(os.getcwd())
def update_config(cfg, yaml_file):
    '''
        This is a function that updates config from yaml file
    '''
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    return cfg
