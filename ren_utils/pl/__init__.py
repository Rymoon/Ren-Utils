
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import skimage
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from typing import List,Tuple,Optional
#
from ren_utils.rennet import get_callable_definition_location,print_WhiteRed_string
# PLOT runner
from pprint import pformat


def configs_by_title(title,*,p_configs,config_override=[]):
    _config_yaml = Path(p_configs)
    assert _config_yaml.suffix == ".yaml"
    assert _config_yaml.exists(), _config_yaml
    with open(_config_yaml,"r") as f:
        d= yaml.load(f,Loader=RenNetLoader)

    configs = {}
    elist= d["experiments"]
    for title_,experiment in elist.items():
        if title_ == title:
            config = experiment["config"]
            compiler = experiment["compiler"]
            for keys,value in config_override:
                _c = config
                _ks = keys.split(".")
                if len(_ks)>1:
                    for k in _ks[:-1]:
                        _c = _c[k]
                _c[_ks[-1]] = value
            configs[title]=  (compiler,config)
            break
    else:
        raise Exception(f"Title {title} not found in {list(elist.keys())}")
    return configs, experiment
    
    
def run_by_title(title, gpuid:int,cfn:str,dm:pl.LightningDataModule,*,compiler_dict:dict,p_configs:str, config_override:List[Tuple[str,Any]]=[]):
    """

    1. config = configs[title]
    2. config === {
        "config_parser": <func name>
        "config": <dict>, # (key:value) --> keyword-args in compiler: compile_model__xxx func.
        ...
    }   
    3. (See `run_configs`)
        - compiler = compiler_dict[compiler_name]
        - trainer, model = call_by_inspect(compiler, config, gpuid = gpuid, cfn = f"{cfn}_{title}",dm=dm)
        - ...

    config_override: For example, [( "dpm_ins.C",3),( "dpm_ins.C",5)]; then ["config"]["dpm_ins"]["C"] is 5
    """
    configs,experiment = configs_by_title(title,p_configs=p_configs,config_override=config_override)
    print(pformat(configs))

    note = {
        "experiment":experiment,    
        "config_override":config_override,
    }
    

    return run_configs(configs, gpuid, cfn, dm,compiler_dict=compiler_dict,note=note)


import yaml
import os
from ren_utils.rennet import call_by_inspect,getitems_as_dict,RenNetDumper,RenNetLoader

def save_yaml(p,d:dict):
    p = Path(p)
    p.parent.mkdir(exist_ok=True,parents=True)
    if p.exists():
        os.remove(p)
    with open(p,"w") as f:
        yaml.dump(d,f,Dumper=RenNetDumper)
    

def load_yaml(p):
    """
    Return None if not exists
    """
    if Path(p).exists():
            
        with open(Path(p),"r") as _f:
            _d = yaml.load(_f,Loader=RenNetLoader)
        return _d
    else:
        return None


def run_configs(configs:dict, gpuid:int,cfn:str,dm:pl.LightningDataModule,*,compiler_dict:dict,note={}):
    """
    compiler_dict:
        - key: parser name
        - value" parser function, (args)-> trainer,model,runner; See functions compile_xxx in `dec.py`.
    
    configs = List[title,[compiler,config]]

    """
    for title,(compiler,config) in configs.items():
        if compiler not in compiler_dict:
            raise KeyError(f"Function `{compiler}` should be found as a key in compiler_dict.")
        config["_compiler"] = compiler
        config["_compiler__setter"] = "ren_utils.pl.run_configs"
        try:
            trainer, model, runner = call_by_inspect(compiler_dict[compiler], config, gpuid = gpuid, cfn = f"{cfn}_{title}",dm=dm)
        except Exception as e:
            print_WhiteRed_string(f'- RenUtilsError: definion of compiler func `{compiler}` at: {get_callable_definition_location(compiler_dict[compiler],format="vst")}')
            raise e
        p = Path(trainer.log_dir,"config.yaml")
        save_yaml(p,config)


        p = Path(trainer.log_dir,"note.yaml")
        save_yaml(p,note)
        
        return runner(trainer, model, dm)

# PLOT callback
# =====================
import time
from datetime import timedelta
from pytorch_lightning.utilities.model_summary import summarize
class ModelSummarySave(pl.Callback):
    def __init__(self,log_dir:str|None=None, fname="model_summary.txt"):
        self.log_dir = log_dir
        self.fname = fname
    
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log_dir = self.log_dir if self.log_dir is not None else trainer.log_dir
        fname = self.fname
        o = summarize(pl_module) # Object ModelSummary
        p=  Path(log_dir,fname)
        
        if p.exists():
            os.remove(p)
        else:
            Path(log_dir).mkdir(parents=True,exist_ok=True)

        msg = str(o)
        with open(p,"x") as f:
            f.write(msg)
        print("- Save model summary:",p.as_posix())
        return super().on_fit_start(trainer, pl_module)

# =====================
class ElapsedTime(pl.Callback):
    def __init__(self):
        self.tic = None
        self.toc = None
        self.fname = "elapsed_time.yaml"
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.tic = time.time()
        return super().on_fit_start(trainer, pl_module)
    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.tic is None:
            self.tic = time.time() # In case you load checkpoint then pass the on_fit_start.
        return super().on_epoch_start(trainer, pl_module)
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epc = trainer.current_epoch
        epm = trainer.max_epochs

        self.toc = time.time()
        print("- elapsed time: ",timedelta(seconds=self.toc-self.tic),"\n")
        if epc!=0:
            print("- remaining time: ",timedelta(seconds=(self.toc-self.tic)/epc*(epm-epc)),"\n")
        return super().on_epoch_end(trainer, pl_module)
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.toc = time.time()
        print("- elapsed time: ",timedelta(seconds=self.toc-self.tic),"\n")

        log_dir =  trainer.log_dir
        fname = self.fname
        d = {
            "human":f"{timedelta(seconds=self.toc-self.tic)}",
            "seconds":f"{self.toc-self.tic}"
        }
        p=  Path(log_dir,fname)

        save_yaml(p,d)
        print("- Save elapsed time:",p.as_posix())
        return super().on_fit_end(trainer, pl_module)
from typing import Callable
def restore(pckpt,pconfig, compiler:Callable,dm, gpuid:int, cfn:str):
    """plog: path to log;
    {plog}/checkpoints/*.ckpt
    {plog}/config.yaml

    compiler: Callable --> trainer, model, runner
    """
    assert Path(pckpt).exists()
    ckpt = torch.load(pckpt.as_posix())

    pconfig = Path(pconfig)
    with open(pconfig,"r") as f:
        config = yaml.load(f,Loader=RenNetLoader)
    
    trainer, model,runner = call_by_inspect(compiler, config, gpuid = gpuid, cfn = cfn,dm=dm)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(f"cuda:{gpuid}"))

    return model, trainer
