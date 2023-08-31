
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

from mha8.utils.rennet import (call_by_inspect, getitems_as_dict)


# PLOT runner
from pprint import pformat
def run_by_title(title, gpuid:int,cfn:str,dm:pl.LightningDataModule,*,compiler_dict:dict,p_configs:str):
    """
    See run_configs
    """
    _config_yaml = Path(p_configs)
    assert _config_yaml.suffix == ".yaml"
    assert _config_yaml.exists(), _config_yaml
    with open(_config_yaml,"r") as f:
        d= yaml.load(f,Loader=RenNetLoader)

    configs = {}
    elist= d["experiments"]
    for title_,v in elist.items():
        if title_ == title:
            configs[title]=  (v["compiler"],v["config"])
            break
    else:
        raise Exception(f"Title {title} not found in {list(elist.keys())}")

    print(pformat(configs))
    

    return run_configs(configs, gpuid, cfn, dm,compiler_dict=compiler_dict)

import yaml
import os
from ren_utils.rennet import call_by_inspect,getitems_as_dict,RenNetDumper,RenNetLoader
def run_configs(configs:dict, gpuid:int,cfn:str,dm:pl.LightningDataModule,*,compiler_dict:dict):
    """
    compiler_dict:
        - key: parser name
        - value" parser function, (args)-> trainer,model,runner; See functions compile_xxx in `dec.py`.
    
    configs = List[title,[compiler,config]]

    """
    for title,(compiler,config) in configs.items():
        if compiler not in compiler_dict:
            raise KeyError(f"Function `{compiler}` should be found as a key in compiler_dict.")
        trainer, model, runner = call_by_inspect(compiler_dict[compiler], config, gpuid = gpuid, cfn = f"{cfn}_{title}",dm=dm)
        p = Path(trainer.log_dir,"config.yaml")
        p.parent.mkdir(exist_ok=True,parents=True)
        if p.exists():
            os.remove(p)
        with open(p,"w") as f:
            yaml.dump(config,f,Dumper=RenNetDumper)
        print(f"- Save yaml: {p}")
        
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
        return super().on_fit_end(trainer, pl_module)