
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
    * <compiler_name> = Loaded yaml-file["compiler"];
    * Find <fcomf> function by <compiler_name> in compiler_dict.
    * <config_d> = yaml-file["compiler"] overrided by <config_override>;
    * Call fcomf(**config_d)
    

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
    
    
# PLOT scan and restore


import os
from pathlib import Path
def collect_expr_paths(root:str,filter_fname_f):
    """
    - filter_fname_f(fname)-->bool
    
    - Skip filtered folder name;
    """
    p = Path(root)
    if not p.exists():
        raise Exception(f"- RuntimeError: PathNotExists: {p}")
    
    fname_l = os.listdir(p)
    fname_l.sort() # Then, same root will have the same order
    
    fname_l_valid = []
    for fname in fname_l:
        if not is_dir(Path(p,fname)):
            continue
        if filter_fname_f(fname):
            fname_l_valid.append(fname)
    
    p_l_valid = [Path(root,fname).as_posix() for fname in fname_l_valid]
    
    return p_l_valid

def is_dir(p):
    try:
        os.listdir(p)
    except Exception as e:
        flag = False
    else:
        flag = True
    return flag



import re
from ren_utils.rennet import call_by_inspect
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import pytorch_lightning as pl
class ResolveDirtree:
    """
    """
    def __init__(self,root):
        self.data = {}
        self.root = root
        if not Path(root).exists():
            raise Exception(f"- RuntimeError: FileNotExists: {root}")
        try:
            os.listdir(root)
        except Exception as e:
            print(f"- RuntimeError: ListDirFailed: May not a valid path to folder. f{root}")
            raise e
        
        
        self.config = None
        self.trainer = None
        self.model = None
        self.runner = None
        
    
    def search_groups(self,pattern,s,n,*,no_except=False):
        """match the pattern of n groups:
        return a list of n element. None if group(i) failed
        """
        try:
            o= re.search(pattern,s)
        except Exception as e:
            if no_except:
                pass
            else:
                print(f"- RuntimeError: RegexFailed (pattern={pattern}): s={s}")
                raise e
        r = []
        for i in range(n):
            try:
                r.append(o.group(i+1))
            except Exception as e:
                if no_except:
                    r.append(None)
                else:
                    print(f"- RuntimeError: \ni+1((i+1)th catched group)={i+1},\nn(number of group to catch)={n},\ns={s}") # 0 is the whole matched
                    raise e
        return r
    
    
    def root_resolve(self,version=0):
        if Path(self.root,"lightning_logs").exists():
            self.root = Path(self.root,"lightning_logs",f"version_{version}").as_posix()
        elif "version" in Path(self.root).stem:
            self.root = Path(Path(self.root).parent,f"version_{version}").as_posix()
        else:
            raise Exception(f"- RuntimeError: root_resolve: {self.root}")
        
    def has_ckpt(self):
        """
        If ckpt saved.
        """
        root = Path(self.root)
        if not Path(root,"checkpoints").exists():
            return False
        else:
            if len(list(os.listdir(Path(root,"checkpoints")))) ==0:
                return False
        
        return True

    def max_ckpt_epoch(self):
        if self.has_ckpt():
            p = Path(self.root,"checkpoints")
            l = os.listdir(p)
            max_ep = -1
            for fname in l:
                epoch = self.search_groups("epoch=([0-9]+)",fname,1,no_except=True)[0]
                if epoch is not None and int(epoch)>max_ep:
                    max_ep = int(epoch)
                    
            if max_ep == -1:
                return None
            else:
                return max_ep
        else:
            return None
    
    def has_tfevent(self):
        """
        """
        root = Path(self.root)
        l = root.glob("events.out.tfevents.*")
        l = list(l)
        if len(l)==0:
            return False
        
        return True
    
    def get_exists_versions(self):
        root = Path(self.root)
        root_v = root
        if "version" in root.stem:
            root_v = root.parent
        elif "lightning_logs" == root.stem:
            root_v = root
        elif "lightning_logs" in list(os.listdir(root)):
            root_v = Path(root,"lightning_logs") 
        
        l = list(os.listdir(root_v))
        versions = []
        for fname in l:
            o = re.match("version_([0-9]+)",fname)
            if o is None:
                continue
            else:
                versions.append(o.group(1))
        versions.sort()
        return versions 
    
    def get_expr_name(self):
        if Path(self.root,"lightning_logs").exists():
            name = Path(self.root).stem
        elif "version" in Path(self.root).stem:
            name = Path(self.root).parent.parent.stem
        else:
            raise Exception(f"- RuntimeError: get_expr_name: self.root={self.root}")
        return name
    
    def get_keys_from_expr_name(self,keys=[]):
        """
        search key from keys, return a list of found keys.
        """
        s = self.get_expr_name()
        l = []
        for k in keys:
            if k in s:
                l.append(k)
        return l
    def get_dataset(self,valid_dataset_names=[]):
        names = self.get_keys_from_expr_name(valid_dataset_names)
        assert len(names)==1, f"-RuntimeError:get_dataset:\n name={names};\n;expr_name={self.get_expr_name()}\nself.root={self.root};\nvalid_dataset_names={valid_dataset_names}"
        return names[0]
    
    def get_score_by_name(self,score_name,save_key=None,sw_fetch_epoch=True): 
        """Process the name of ckpts.
        value of score is a float, matching ([0-9.]+)
        if save_key is None, then make the same as score_name.
        """
        root = Path(self.root)
        p = Path(root,"checkpoints")
        save_key = score_name if save_key is None else save_key
        if p.exists():
            l = list(p.glob("*.ckpt"))
            for fpath in l:
                score = self.search_groups(f"{score_name}=([0-9.]+)",fpath.stem,1,no_except=True)[0]
                if sw_fetch_epoch:
                    epoch = self.search_groups("epoch=([0-9.]+)",fpath.stem,1,no_except=True)[0]
                if score is not None and epoch is not None:
                    score = float(score)
                    epoch = int(epoch)
                    break
            if sw_fetch_epoch:
                return {
                    "_key":save_key,
                    "epoch":epoch,
                    "value":score,
                    "path":fpath.as_posix() if score is not None else None
                }
            else:
                return {
                    "_key":save_key,
                    "value":score,
                    "path":fpath.as_posix() if score is not None else None
                }
        else:
            if sw_fetch_epoch:
                return  {
                    "_key":save_key,
                    "epoch":None,
                    "value":None,
                    "path":None
                }
            else:
                return  {
                    "_key":save_key,
                    "value":None,
                    "path":None
                }

    def get_ckpts_lazy(self):
        """
        yield dict.
        
        d[path]: path to ckpt
        d[epoch]: when it stop
        d[tag]: why it stop
        d[tag2]: why it stop details
        d[..] some other things
        """
        root = Path(self.root)
        p = Path(root,"checkpoints")
        d = {
        }
        if p.exists():
            l = list(p.glob("*.ckpt"))
            for fpath in l:
                d = {
                    "_key":"ckpt",
                    "tag":None,
                    "path":fpath.as_posix(),
                }
                if "last" in fpath.stem:
                    if "epoch" in fpath.stem:
                        epoch = self.search_groups("epoch=([0-9]+)",fpath.stem,1)[0]
                        d.update( {
                            "tag":"last",
                            "epoch":int(epoch) if epoch is not None else None,
                            })
                    else:
                        d.update( {
                            "tag":"last",
                            "epoch":None
                            })
                    
                yield d
    
    def get_tfevent(self,scalar_keys=[],as_value_fs=[]):
        l = Path(self.root).glob("events.out.tfevents.*")
        l = list(l)
        d = {}
        if len(l)==0:
            return {}
        elif len(l)==1:
            p = Path(l[0])
            d["_key"] = "tfevent"
            d["path"] = p.as_posix()
            # Create an EventAccumulator object
            event_acc = EventAccumulator(p.as_posix())
            event_acc.Reload()
            d["scalar_available_keys"] = event_acc.Tags()["scalars"]
            d["scalars"] = {}
            d["scalars"]["epoch"] = []
            from datetime import datetime
            from collections import namedtuple
            
            for i in range(len(scalar_keys)):
                key = scalar_keys[i]
                as_value_f = as_value_fs[i]
                if key in event_acc.Tags()['scalars']:
                    events = event_acc.Scalars(key)

                    # Extract the steps and loss values
                    d_ = {
                            "_key":key,
                            "time_str":[],
                            "wall_time":[],
                            "value":[],
                            "step":[]
                        }
                    for se in events:
                        dt = datetime.fromtimestamp(se.wall_time)
                        d_ ["time_str"].append(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")
                        d_["wall_time"].append(se.wall_time)
                        d_["value"].append(as_value_f(se.value))
                        d_["step"].append(se.step)
                else:
                    d_ = {
                        "_key":key,
                        "value":None,
                    }
                d[key] = d_        
        else:
            raise Exception(f"- RuntimeError: MultipleTfevents: {l}")            

        return d
    
    def get_config(self):
        import mha8.apps.DecNet.dec_running_h # will add things to RenNetLoader
        p = Path(self.root,"config.yaml")
        if not p.exists():
            return None
        else:
            with open(p.as_posix(),"r") as f:
                s = f.read()
            d = yaml.load(s,Loader=RenNetLoader)
            return d
    
    def get_note(self):
        p = Path(self.root,"note.yaml")
        if not p.exists():
            return None
        else:
            with open(p.as_posix(),"r") as f:
                s = f.read()
            d = yaml.load(s,Loader=RenNetLoader)
            return d
    
    def load_config(self):
        self.config = self.load_config()
        if self.config is None:
            raise Exception(f"- RuntimeError: config is None ")
    def load_note(self,no_except=False):
        self.note = self.get_note()
        if (not no_except) and self.note is None:
            raise Exception(f"- RuntimeError: note is None")
    
    def load_model(self,*,comp_env_kwargs:dict={},compiler_d:dict,cfn="collect_default"):
        """
        comp_env_kwargs ={
            gpuid:int,
            cfn:str,
            ...
        }
        """
        self.load_config()
        config = self.config
        
        # See ren_utils.pl.__init__.py:run_configs for more details.
        compiler_name = config["experiment"]["compiler"]
        try:
            compiler_f = compiler_d[compiler_name] 
        except Exception as e:
            raise e

        trainer, model, runner = call_by_inspect(compiler_f,config,**comp_env_kwargs)
        
        self.trainer = trainer
        self.model = model
        self.runner = runner
        
from tqdm import tqdm,trange
def walk_in_logs(p_results,filter_f=lambda _: True):
    """
    Yield resolver
    
    Have a trange inside!
    """
    e_p_list = collect_expr_paths(p_results,filter_f)
    for i in trange(len(e_p_list)):
        e_root = Path(e_p_list[i])
        rsvr = ResolveDirtree(e_root)
        ver_list = rsvr.get_exists_versions()
        for ver in ver_list:
            rsvr.root_resolve(ver)
            yield rsvr
            

from ren_utils.rennet import call_by_inspect,getitems_as_dict,get_root_Results
import torch
from typing import Optional,List,Tuple,Dict
class RestoreModel:
    
    def __init__(self):
        """
        """
        self.compiler = None
        self.config = None
        self.model = None # type: Optional[torch.nn.Module]
        self.trainer = None
        self.runner = None
        self.gpuid = None
        self.ckpt_path = None
        
    
    def set_compiler_by_name(self,name,py_script):
        """
        py_script: the `.py` file contained the compiler object/function.
        """
        try:
            compiler = getattr(py_script,name)
        except Exception as e:
            
            raise e
        self.compiler = compiler
        
    def set_compiler_by_config(self,config=None,py_script=None):
        assert py_script is not None
        if config is None:
            config = self.config
        else:
            self.config = config
        
        if "_compiler" in config:
            self.set_compiler_by_name(config["_compiler"],py_script)
        else:
            raise Exception(f"- RuntimeError: Might be old version experiment logs. Please use Manager.collect.scan_v0.6::add_compiler_key_to_config.")
    
    def compile(self,comp_env_kwargs:dict):
        """
        Requires:
        - self.config
        - self.compiler
        
        
        Return trainer, model, runner
        """
        assert "gpuid" in comp_env_kwargs
        gpuid = comp_env_kwargs["gpuid"]
        assert "cfn" in comp_env_kwargs
        assert self.compiler is not None, f"- RuntimeError: self.compiler is None: use self.set_compiler_by_name or set_compiler_by_config."
        assert self.config is not None, f"- RuntimeError: self.config is None."
        trainer, model,runner  = call_by_inspect(self.compiler,self.config,**comp_env_kwargs)
        self.trainer = trainer
        self.model = model
        self.runner = runner
        self.gpuid = gpuid
        return trainer, model, runner
        
    def compile_model(self,comp_env_kwargs:dict):
        """
        Return model
        """
        trainer, model, runner = self.compile(comp_env_kwargs)
        return model

    def load_ckpt(self,gpuid,rsv:ResolveDirtree,ckpt_filter_f):
        assert self.model is not None
        checked_ckpt_paths = []
        for d in rsv.get_ckpts_lazy():
            p2 = Path(d["path"])
            assert p2.exists(),p2
            checked_ckpt_paths.append(p2.as_posix())
            if ckpt_filter_f(p2.as_posix()):
                break
        else:
            raise Exception(f"- RuntimeError: NoValidCkpt:\n- expr_path: {rsv.root}\n- checked_ckpt_paths={checked_ckpt_paths}\n- ckpt_filter_f: {ckpt_filter_f} ")
        ckpt_blob = torch.load(p2.as_posix(),map_location=torch.device(f"cuda:{gpuid}"))
        self.ckpt_path = p2.as_posix()
        self.model.load_state_dict(ckpt_blob["state_dict"])
        self.model.to(torch.device(f"cuda:{gpuid}"))
        
def load_models(rsv_list:List[ResolveDirtree],with_ckpt=False,*,ckpt_filter_f=None,py_script=None,config_override=[],comp_env_kwargs={}):
    """
    - ckpt_filter_f(fullpath_to_ckpt)-> bool
    config_override = [(key,value),(key_list,value),...]
    Return: List of RestoreModel
    """
    return list(load_models_lazy(rsv_list,with_ckpt,ckpt_filter_f=ckpt_filter_f,py_script=py_script,config_override=config_override,comp_env_kwargs=comp_env_kwargs))

def load_models_lazy(rsv_list:List[ResolveDirtree],with_ckpt=False,*,ckpt_filter_f=None,py_script=None,config_override=[],comp_env_kwargs={}):
    """
    - ckpt_filter_f(fullpath_to_ckpt)-> bool
    config_override = [(key_l,value),...]
    key_l = [key,...] # if is List (Not tuple!)
    Return:  Iterator of RestoreModel
    """
    assert py_script is not None
    assert "gpuid" in comp_env_kwargs
    gpuid = comp_env_kwargs["gpuid"]
    for rsv in rsv_list:
        config = rsv.get_config()
        if len(config_override)>0:
            for ovr in config_override:
                if isinstance(ovr[0],list) or isinstance(ovr[0],tuple):
                    c = config
                    for i_ in range(len(ovr[0])-1):
                        c=c[ovr[0][i_]]
                    k = ovr[0][-1]
                else:
                    c = config
                    k = ovr[0]
                c[k] = ovr[1]
        else:
            pass
        rm  = RestoreModel()
        rm.set_compiler_by_config(config,py_script)
        rm.compile_model(comp_env_kwargs)
        
        if with_ckpt:
            rm.load_ckpt(gpuid,rsv,ckpt_filter_f)
        
        yield rm
        
            
            
from typing import Callable
def restore(pckpt,pconfig, compiler:Callable,dm, gpuid:int, cfn:str):
    """plog: path to log;
    {plog}/checkpoints/*.ckpt
    {plog}/config.yaml

    compiler: Callable --> trainer, model, runner
    """
    if pckpt is not None:
        assert Path(pckpt).exists()
        ckpt = torch.load(pckpt.as_posix())
    else:
        ckpt = None

    pconfig = Path(pconfig)
    with open(pconfig,"r") as f:
        config = yaml.load(f,Loader=RenNetLoader)
    
    trainer, model,runner = call_by_inspect(compiler, config, gpuid = gpuid, cfn = cfn,dm=dm)
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(f"cuda:{gpuid}"))

    return model, trainer
