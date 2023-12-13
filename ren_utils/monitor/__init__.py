
import torch
import json
from pathlib import Path,PosixPath
import shutil
import os
from typing import List,Tuple,Dict,overload,Any,Optional
from pprint import pprint,pformat
from ren_utils.rennet import getTimestrAsFilename,RenNetJSONEncoder,JSONDict
import numpy as np



class Summary(JSONDict):
    def __init__(self,errors:dict={},missings ={},extras={}):
        JSONDict.__init__(self)
        self.errors = errors   
        self.missings = missings
        self.extras = extras
        
    def as_dict(self):
        return {
            "errors":self.errors, # pindex:info
            "missings":self.missings,
            "extras":self.extras,
        }
    def __str__(self):
        msg = f"Summary(n_errors:{self.n_errors},n_missings:{self.n_missings},n_extras:{self.n_extras})"
        if self.n_errors<5:
            msg+="\nerror:\n"+pformat(self.errors)
            msg+="\nmissing:\n" +pformat(self.missings)
            msg+="\nextra:\n"+pformat(self.extras)
        return msg
    
    @property
    def n_errors(self):
        return len(self.errors)
    
    @property
    def n_missings(self):
        return len(self.missings)
    
    @property
    def n_extras(self):
        return len(self.extras)
    
    
class RunTracer:
    def __init__(self,output_dir:str,input_dir:Optional[str]=None):
        self.output_dir = output_dir 
        self.input_dir = input_dir
        #
        self.cnt_state_calling = 0
        self.states = {
        }
        
        print(f"- Warning: RunTracer class: Currently, n_missing or info:#InOnly is useless. Since result is generate by state calling and wont de;iberately scan all the things in input_dir.")
    
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
        _print = lambda*args: print(f"- RT: [{self.cnt_state_calling:03}] ",*args)
        _get_fp = lambda k:Path(self.output_dir,f"{k}.state")
        _p = _get_fp(key)
        _v = value
        if Path(_p).exists():
            i = 1
            old_key = key
            _get_key = lambda i: f"{old_key}___{i:03}"
            key = _get_key(i)
            _p = _get_fp(key)
            while _p.exists():
                i=i+1
                key = _get_key(i)
                _p = _get_fp(key)
            _print(f"RuntimeInfo: RunTracer: StateDuplicateKey({old_key}), new_key={key}")
            _print(f"Save state[{key}]: {_p}")
            
        else:
            _print(f"Save state[{key}]: {_p}")
        output_d = {
            "key":key,
            "value":_v
        }
        torch.save(output_d,_p)
        #
        result = None
        summary = None
        if self.input_dir is not None:
            m = list(Path(self.input_dir).glob(f"{key}.state"))
            if len(m)==0:
                result = {
                    "___ret_type":"Error",
                    "info":"#OnlyOut",
                    "pindex":Path(key).as_posix(),
                    "as_bool":True,
                }
                _print(f"MissingInputState, key= {key}")
            elif len(m)>1:
                result = {
                    "___ret_type":"Error",
                    "info":"#MultipleIn",
                    "pindex":Path(key).as_posix(),
                    "as_bool":False,
                }
                _print(f"RuntimeError: DuplicateInputState, matched: {m}")
            else:
                _p_in = m[0]
                _print(f"Input state[{key}]: {_p_in}")
                
                input_d = torch.load(Path(_p_in).as_posix())
                result = self.compare(Path(key),output_d, input_d)
                
                summary = self.summarize(result)
                _print(f"matched? : {summary}")
                
        self.states[key] = {
            "order":self.cnt_state_calling,
            "result":result,
            "summary":summary,
        }
        self.cnt_state_calling+=1
    
    def summarize(self,result):
        summary = Summary()
        self._summarize(summary,result)
        
        return summary
    
    def _summarize(self,summary:Summary,result):
        """
        check ["as_bool"]
        
        skip key startwith "___" in apply summarize_
        """
        if self.is_final(result):
            if not result["as_bool"]:
                summary.errors[result["pindex"]] = result["info"]
            if result["___ret_type"] == "Error" and result["info"]=="#InOnly":
                summary.missings[result["pindex"]] = result["type"]["in"]
            if result["___ret_type"] == "Error" and result["info"] == "#OutOnly":
                summary.extras[result["pindex"]] = result["type"]["out"]
            return 
        else:
            for k,v in result.items():
                if k[:3] == "___":
                    continue
                self._summarize(summary,v)
            return 
    @staticmethod
    def is_final(d:dict):
        return "___ret_type" in d
    def compare(self,root,output_d,input_d):
        assert output_d["key"] == input_d["key"]
        result = self._compare(root,output_d["value"],input_d["value"])
        return result
    
    @staticmethod
    def _compare_valueType(out_value,in_value):
        try:
            eq = out_value == in_value
            info = "#=="
            as_bool = eq
        except Exception as e:
            eq = None
            info = "#Exception(==): {e}"
            as_bool = False
        return  {"info":info,"equal":eq,"as_bool":as_bool}
    
    def _compare(self,root,out_value,in_value):
        """
        - Skip key "___xxx" in a dict;
        - Indices of list, will store as str in result as key;
        - All result entities are dict;
        - result returned can be an empty {}
        """
        if isinstance(out_value,(str,bool)):
            result = {
                "___ret_type":"ValueType","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),
                "in_value":in_value,
                "out_value":out_value,
            }
            if isinstance(in_value,PosixPath):
                in_value =in_value.as_posix()
            result.update(self._compare_valueType(out_value,in_value))
            
            
            return result
        # Path
        elif isinstance(out_value,PosixPath):
            result = {
                "___ret_type":"ValueType","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),
                "in_value":in_value,
                "out_value":out_value,
            }
            out_value= out_value.as_posix()
            if isinstance(in_value,PosixPath):
                in_value =in_value.as_posix()
            result.update(self._compare_valueType(out_value,in_value))
            return result
        # MappingContainer
        elif isinstance(out_value,dict):
            if isinstance(in_value,dict):
                result = {}
                for k,v in out_value.items():
                    if k[:3]=="___":
                        continue
                    if k in in_value.keys():
                        result[k] = self._compare(Path(root,k),v,in_value[k])
                    else:
                        result[k] = {"___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"info":"#OutOnly","pindex":Path(root,k).as_posix(),"as_bool":True}
                for ki,vi in in_value.items():
                    if ki not in out_value:
                        result[ki] = {"___ret_type":"Error","type":{"out":None,"in":type(in_value).__name__},"pindex":Path(root,ki).as_posix(),"info":"#InOnly","as_bool":True}
            else:
                result = {
                    "___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),"info":"#MismatchDictType","as_bool":False}
        # SeriesContainer
        elif isinstance(out_value,(list,tuple)):
            if isinstance(in_value,(list,tuple)):
                result = {}
                for i,v in enumerate(out_value):
                    if i< len(in_value):
                        result[str(i)] =(self._compare(Path(root,str(i)),v,in_value[i]))
                    else:
                        result[str(i)] ={
                            "___ret_type":"Error","type":{"out":type(out_value).__name__,"in":None},
                            "pindex":Path(root,f"{i}").as_posix(),
                            "info":"#OutOnly","as_bool":True
                            }
                else:
                    if i <len(in_value)-1:
                        i=i+1
                        while i<len(input_value):
                            result[str(i)]={"___ret_type":"Error","type":{"out":None,"in":type(in_value).__name__},"pindex":Path(root,f"{i}").as_posix(),"info":"#InOnly","as_bool":True}
                            i=i+1
            else:
                result = {"___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),"info":"#MismatchListType","as_bool":False}
        # Float/Matrix
        elif isinstance(out_value,(int,float,torch.Tensor,np.ndarray)) or isinstance(in_value,(int,float,torch.Tensor,np.ndarray)):
            result = {
                "___ret_type":"Numerical",
                "type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),
                "info":"torch.equal"
            }
            if isinstance(out_value,(torch.Tensor,int,float,np.ndarray)) and isinstance(in_value,(torch.Tensor,int,float,np.ndarray)):
                if isinstance(out_value, (float,int)):
                    result["out_shape"] = None
                elif isinstance(out_value,np.ndarray):
                    out_value = torch.from_numpy(out_value)
                    result["out_shape"] = out_value.shape
                else: # Tensor
                    out_value = out_value.cpu()
                    result["out_shape"] = out_value.shape

                if isinstance(in_value,(float,int)):
                    result["in_shape"] = None
                elif isinstance(in_value,np.ndarray):
                    in_value = torch.from_numpy(in_value)
                    result["in_shape"] = in_value.shape
                else:
                    in_value = in_value.cpu()
                    result["in_shape"] = in_value.shape
                
                if result["in_shape"] is None:
                    result["in_value"] = in_value
                    if result["out_shape"] is not None:
                        in_value = in_value * torch.ones_like(out_value)
                    else:
                        result["out_value"] = out_value
                        in_value = torch.tensor(in_value)
                        out_value = torch.tensor(out_value)
                elif result["out_shape"] is None:
                    result["out_value"] = out_value
                    out_value = out_value.torch.ones_like(in_value)
                    
                try:
                    result["dist_fro"] = torch.norm(in_value.float()-out_value.float(),p="fro")
                except Exception as e:
                    result["dist_fro"] = f"#Exception: {e}"
                    
                try:
                    result["mse_loss"] = torch.nn.functional.mse_loss(in_value.float(),out_value.float()),
                except Exception as e:
                    result["mse_loss"] = f"#Exception: {e}"
                
                try:
                    result["equal"] = torch.equal(out_value,in_value)
                except Exception as e:
                    result["equal"] = f"#Exception: {e}"
                
                if isinstance(result["equal"],bool):
                    result["as_bool"] = result["equal"]
                else:
                    result["info"]="#EqualOperatorFailed"
                    result["as_bool"] = False
            else:
                result["info"] = "#MismatchNumericalType"
                result["as_bool"] = False
        else:
            result = {
                "___ret_type":"Error",
                "type":{"out":type(out_value).__name__,"in":type(in_value).__name__},
                "pindex":Path(root).as_posix(),
                "info":"#NotImplementType",
                "as_bool":False
            }
        return result
    
    def summarize_states(self): 
        """
        return summary,all_results
        """
        all_results = {}
        for key,v in self.states.items():
            if v["result"] is not None:
                
                all_results[key] = {
                    "___order":v["order"] # to skip comparison
                }
                all_results[key].update(v["result"])
        summary = self.summarize(all_results)
        return summary,all_results

    
    def save_summary(self,summary:Summary,result:Optional[dict]):
        """If the result provided, then a detailed version will be saved.
        """
        
        p=Path(self.output_dir,f"summary-{getTimestrAsFilename()}.json")
        
        if result is None:
            s = json.dumps(summary,cls=RenNetJSONEncoder)
        else:
            d = {}
            for pindex in summary.errors.keys():
                r = result
                for key in Path(pindex).parts:
                    r = r[key]
                d[pindex] = r
            s = json.dumps(d,cls=RenNetJSONEncoder)
        with open(p,"w") as f:
            f.write(s)
        print(f"- Save json: {p.as_posix()}")
        
        
import ren_utils
ren_utils.global_objects["RunTracer"] = None
def globalRunTracer():
    if ren_utils.global_objects["RunTracer"] is None:
        ren_utils.global_objects["RunTracer"] = RunTracer(
            output_dir = None,
            input_dir = None
        )
    return ren_utils.global_objects["RunTracer"]