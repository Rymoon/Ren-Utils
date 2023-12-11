
import torch
import json
from pathlib import Path
import shutil
import os
from typing import List,Tuple,Dict,overload,Any,Optional
class RunTracer:
    def __init__(self,output_dir:str,input_dir:Optional[str]=None):
        self.output_dir = output_dir 
        self.input_dir = input_dir
        #
        self.cnt_state_calling = 0
    
    @property
    def output_dir(self):
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self,p):
        if p is not None:
            p = Path(p).as_posix()
            if Path(p).exists():
                shutil.rmtree(p)
            Path(p).mkdir(parents=True)
            self._output_dir = p
        else:
            self._output_dir = None
    
    @property
    def input_dir(self):
        return self._input_dir
    @input_dir.setter
    def input_dir(self,p):
        if p is not None:
            if not Path(p).exists():
                raise Exception(f"- RuntimeError: PathNotExists: input_dir={p}")
            else:
                p= Path(p).as_posix()
                self._input_dir = p
        else:
            self._input_dir = None
            
        
    def state(self,key,value):
        if self.output_dir is None:
            raise Exception(f"- NotInitializedError: Assign self.output_dir before calling self.state(...).")
        _print = lambda*args: print(f"- [{self.cnt_state_calling:03}] ",*args)
        _p = Path(self.output_dir,f"{key}.state")
        _v = value
        if Path(_p).exists():
            i = 1
            _pp = Path(self.output_dir,f"{key}___{i:03}.state")
            while _pp.exists():
                i=i+1
            _p = _pp
            _print(f"RuntimeError: RunTracer: DuplicateOutputState, key={key}; Renamed as: {_p}")
        else:
            _print(f"Save state[{key}]: {_p}")
        output_d = {
            "key":key,
            "value":_v
        }
        torch.save(output_d,_p)
        
        if self.input_dir is not None:
            m = list(Path(self.input_dir).glob(f"{key}.state"))
            if len(m)==0:
                _print(f"MissingInputState, key= {key}")
            elif len(m)>1:
                _print(f"RuntimeError: DuplicateInputState, matched: {m}")
            else:
                _p_in = m[0]
                _print(f"Input state[{key}]: {_p_in}")
                
                input_d = torch.load(Path(_p_in).as_posix())
                comp = self.compare(output_d, input_d)
                _print(f"matched? : {comp}")
                
        self.cnt_state_calling+=1
        
    def compare(self,output_d,input_d):
        assert output_d["key"] == input_d["key"]
        self._compare(Path("/"),output_d["value"],input_d["value"])
        
    
    def _compare(self,root,out_value,in_value):
        if isinstance(out_value,dict):
            if isinstance(in_value,dict):
                result = {}
                for k,v in out_value.items():
                    if k in in_value.keys():
                        result[k] = self._compare(Path(root,k),v,in_value[k])
                    else:
                        result[k] = {"___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"as_bool":True,"info":"#OutOnly"}
                for ki,vi in in_value.items():
                    if ki not in out_value:
                        result[ki] = {"___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"as_bool":True,"info":"#InOnly"}
            else:
                result = {
                    "___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"as_bool":False,
                    "info":"#MismatchContainerType"}
        elif isinstance(out_value,(list,tuple)):
            if isinstance(in_value,(list,tuple)):
                result = []
                for i,v in enumerate(out_value):
                    if i< len(input_value):
                        result.append(self._compare(Path(root,i),v,input_value[i]))
                    else:
                        result.append({
                            "___ret_type":"Error",
                            "info":"#OutOnly"
                            })
                else:
                    if i <len(input_value)-1:
                        i=i+1
                        while i<len(input_value):
                            result.append({"___ret_type":"Error","info":"#InOnly"})
                            i=i+1
            else:
                result = {"___ret_type":"Error","info":f"#Out({type(out_value)})|In({type(in_value)})"}
        elif isinstance(out_value,(int,float,torch.Tensor)):
            result = {
                "___ret_type":"Numerical",
                "info":f"#Out({type(out_value)})|In({type(in_value)})"
            }
            if isinstance(out_value, (float,int)):
                result["out_shape"] = None
            else:
                result["out_shape"] = out_value.shape
            
            if isinstance(in_value,(float,int)):
                result["in_shape"] = None
            else:
                result["in_shape"] = in_value.shape
            
            if result["in_shape"] is None:
                if result["out_shape"] is not None:
                    in_value = in_value * torch.ones_like(out_value)
                else:
                    in_value = torch.tensor(in_value)
                    out_value = torch.tensor(out_value)
            elif result["out_shape"] is None:
                out_value = out_value.torch.ones_like(in_value)
            elif in_value.shape == out_value.shape:
                result.update({
                    "dist_fro":torch.norm(in_value-out_value,p="fro"),
                    "mse_loss":torch.nn.functional.mse_loss(in_value,out_value),
                    "equal":torch.equal(out_value,in_value),
                })
                
            else:
                result = {"___ret_type":"Error","info":f"#Out({type(out_value)})|In({type(in_value)})"}

        elif isinstance(out_value,(int,float,str,bool)):
            result = {
                "___ret_type":"ValueType",
                "type":{"out":type(out_value).__name__,"in":type(in_value).__name__},
                "as_bool": out_value == type(out_value)(in_value),
                "info":"#EqaulOperator"
            }
        else:
            result = {
                "___ret_type":"Error",
                "info":f"#Out({type(out_value)})|In({type(in_value)})",
            }
        return result
            
import ren_utils
ren_utils.global_objects["RunTracer"] = None
def globalRunTracer():
    if ren_utils.global_objects["RunTracer"] is None:
        ren_utils.global_objects["RunTracer"] = RunTracer(
            output_dir = None,
            input_dir = None
        )
    return ren_utils.global_objects["RunTracer"]