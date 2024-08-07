
import torch
from typing import (Generic, Iterable, List, Optional, Tuple, TypeVar, Union,
                    overload)
# PLOT Data structure, data easy-fetch
_T_co= TypeVar('_T_co')
class batch(Generic[_T_co]):
    """Iterator.
    Yield tuple of length sz_batch.
    The last yielded-value may smaller than sz_btach.
    """
    def __init__(self,it:Iterable[_T_co],sz_batch:int):
        self.it = it
        self.sz_batch = sz_batch

    def __iter__(self):
        while True:
            cache =tuple(obj for i,obj 
                in zip(range(self.sz_batch),self.it))
            if len(cache)==0:
                #raise StopIteration()
                return
            yield cache


class LazyList:
    """
    A list of callable things.
    
    [f0,f1,...]

    lazydict[0] --> f0(0)
    
    
    
    
    .. note::
        **When to use?**
    
        We will lose references if store buffer-tensors in built-in list, when
            - Use `model.to(device)` to move all buffer-tensors. 
        
        PyTorch has registered `Parameter` to avoid such reference loss, but no `Buffer` class.
        
        See Parameter and ParamterList
            
        `LazyList,LazyDict` will call getattr immediately when getitem. 
    """
    def __init__(self,value):
        self.data = list(value)
    def __getitem__(self,key):
        f = self.data[key]
        return f(key)
    def __len__(self):
        return len(self.data)
    def __setitem__(self,key,value):
        self.data[key] = value

class LazyDict:
    """
    A dict of Callable things.

    {key:fkey,...}

    lazydict[key] --> fkey(key)
    
    .. note::
        
        **When to use?**
    
        See class `LazyList`
    """

    def __init__(self,value):
        self.data = dict(value)
    def __getitem__(self,key):
        f = self.data[key]
        return f(key)
    def __len__(self):
        return len(self.data)
    def __setitem__(self,key,value):
        self.data[key] = value

from typing import Dict
class BufferDict(torch.nn.Module):
    def __init__(self,d:Dict,persistent:bool):
        super().__init__()
        for k,v in d.items():
            self.register_buffer(k,v,persistent=persistent)
        
# PLOT config
import json
from pathlib import PosixPath


from ren_utils.rennet import MsgedDict,JSONDict,JSONList
class RenNetJSONEncoder(json.JSONEncoder):
    """
    - MsgedDict: Dict
    - PosixPath: str
    """
    def default(self, obj):
        if isinstance(obj, MsgedDict):
            return dict(obj)
        elif isinstance(obj,PosixPath):
            return obj.as_posix()
        elif isinstance(obj,JSONDict):
            return obj.as_dict()
        elif isinstance(obj,JSONList):
            return obj.as_list()
        elif isinstance(obj,torch.Tensor):
            if obj.numel() == 1:
                obj =  obj.detach().cpu()
                return obj.item()
            else:
                return f"TensorShape({obj.shape})"
        else:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)
