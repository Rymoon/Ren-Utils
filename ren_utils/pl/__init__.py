
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