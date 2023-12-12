
import torch
import json
from pathlib import Path,PosixPath
import shutil
import os
from typing import List,Tuple,Dict,overload,Any,Optional
from pprint import pprint,pformat
from ren_utils.rennet import getTimestrAsFilename,RenNetJSONEncoder,JSONDict



class Summary(JSONDict):
    def __init__(self,errors:dict={}):
        JSONDict.__init__(self)
        self.errors = errors   
        
    def as_dict(self):
        return {
            "errors":self.errors # pindex:info
        }
    def __str__(self):
        msg = f"Summary(n_errors:{self.n_errors})"
        if self.n_errors<5:
            msg+="\n"+pformat(self.errors)
        return msg
    
    @property
    def n_errors(self):
        return len(self.errors)
    
    
class RunTracer:
    def __init__(self,output_dir:str,input_dir:Optional[str]=None):
        self.output_dir = output_dir 
        self.input_dir = input_dir
        #
        self.cnt_state_calling = 0
        self.states = {
        }
    
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
        result = None
        summary = None
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
                "___ret_type":"ValueType","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix()
            }
            if isinstance(in_value,PosixPath):
                in_value =in_value.as_posix()
            result.update(self._compare_valueType(out_value,in_value))
            return result
        elif isinstance(out_value,PosixPath):
            result = {
                "___ret_type":"ValueType","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix()
            }
            out_value= out_value.as_posix()
            if isinstance(in_value,PosixPath):
                in_value =in_value.as_posix()
            result.update(self._compare_valueType(out_value,in_value))
            return result
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
        elif isinstance(out_value,(list,tuple)):
            if isinstance(in_value,(list,tuple)):
                result = {}
                for i,v in enumerate(out_value):
                    if i< len(input_value):
                        result[str(i)] =(self._compare(Path(root,i),v,input_value[i]))
                    else:
                        result[str(i)] ={
                            "___ret_type":"Error","type":{"out":type(out_value).__name__,"in":None},
                            "pindex":Path(root,f"{i}").as_posix(),
                            "info":"#OutOnly","as_bool":True
                            }
                else:
                    if i <len(input_value)-1:
                        i=i+1
                        while i<len(input_value):
                            result[str(i)]={"___ret_type":"Error","type":{"out":None,"in":type(in_value).__name__},"pindex":Path(root,f"{i}").as_posix(),"info":"#InOnly","as_bool":True}
                            i=i+1
            else:
                result = {"___ret_type":"Error","type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),"info":"#MismatchListType","as_bool":False}
        elif isinstance(out_value,(int,float,torch.Tensor)) or isinstance(in_value,(int,float,torch.Tensor)):
            result = {
                "___ret_type":"Numerical",
                "type":{"out":type(out_value).__name__,"in":type(in_value).__name__},"pindex":Path(root).as_posix(),
                "info":"torch.equal"
            }
            if isinstance(out_value,(torch.Tensor,int,float)) and isinstance(in_value,(torch.Tensor,int,float)):
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
                
                result.update({
                    "dist_fro":torch.norm(in_value.float()-out_value.float(),p="fro"),
                    "mse_loss":torch.nn.functional.mse_loss(in_value.float(),out_value.float()),
                    "equal":torch.equal(out_value,in_value),
                })
                
                result["as_bool"] = result["equal"]
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