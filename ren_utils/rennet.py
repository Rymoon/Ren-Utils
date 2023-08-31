# Something not depend on pytorch 
# From RenNet, 23-04-30

import os
from pathlib import Path
import json

'''
From RenNet/env.py,



Example of apps/rennet.json:
{
    "root_Results":"/.../Results",
    "pcache_invK":"/.../Results/pcache_invK/",
    "datasets":{
        "CelebAHQ256":{
            "imgs":"/.../Dataset/CelebAHQ/data256x256/",
            "suffix":"jpg"
            },
        "CelebAHQ256_valid":{
            "imgs":"/.../Dataset/CelebAHQ/data256x256_valid/",
            "suffix":"jpg"
        },
        "CelebAHQ256_1":{
            "imgs":"/.../Dataset/CelebAHQ_1/",
            "suffix":"jpg"
        },
        "CelebAHQ256_2":{
            "imgs":"/.../Dataset/CelebAHQ_2/",
            "suffix":"jpg"
        }
    }
}

'''
with open(Path(Path(__file__).parent,f"{Path(__file__).stem}.json").as_posix()) as _f:
    _d=  json.loads(_f.read())

root_Results=  _d["root_Results"]
datasets = _d["datasets"] 
'''
2021-2-5, RenNet/framework/Core/RyCore/__init__
'''
from datetime import datetime
import hashlib
import regex
import inspect
import json
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import currentframe, getframeinfo, signature
from typing import (Generic, Iterable, List, Optional, Tuple, TypeVar, Union,
                    overload)


# from RenNet.framework.Core import console
# print = console.log

def EMPTY_FUNCTION(*args,**kargs):
    pass
ENDL = "\n"
#


# PLOT Console

from enum import Enum


class Color(Enum):
    red=  'red'
    green = 'green'

def COLOR(s:str,color:Color=Color.red)->str:
    '''
    r   red
    g   green
    '''
    if color == Color.red:
        r= '\033[5;31;40m{}\033[0m'.format(s)
    elif color ==Color.green:
        r = '\033[4;32;40m{}\033[0m'.format(s)
    else:
        r = s
    return r



def print_notice(*obj_list):
    """
    Green"""
    print(*obj_list,style="green")

def print_error(*obj_list):
    """
    Red"""
    print(*obj_list,style="red")


# PLOT Time
def sec2hr(sec:float):
    """
    h,m,s: float,float,float
    """
    h = sec//60//60
    m = (sec//60)%60
    s = sec%60
    return h,m,s

def getTimestr():
    '''
    2021-02-08T11:56:15.762
    '''
    
    now = datetime.now() #obj_like datetime(2022, 3, 22, 15, 36, 13, 369673)
    now_str = f"{now.year}-{now.month:0>2d}-{now.day:0>2d}-T{now.hour:0>2d}:{now.minute:0>2d}:{now.second:0>2d}.{now.microsecond//1000:0>3d}"
    
    return now_str

def getTimestrAsFilename():
    """
    yyyy-MM-dd-hh-mm-ss-zzz
    """
    now = datetime.now() #obj_like datetime(2022, 3, 22, 15, 36, 13, 369673)
    now_str = f"{now.year}-{now.month:0>2d}-{now.day:0>2d}-{now.hour:0>2d}-{now.minute:0>2d}-{now.second:0>2d}-{now.microsecond//1000:0>3d}"
    
    return now_str


def getcfp(s:str):
    p = os.path.split(s)[0]
    return p



class Errmsg:
    """errmsg_data and errmsg_append. A mixin class.
    
    .errmsg_data = [msg]
    - msg = (title:str,cont:str)

    If (N:=max_n_errmsg)>0:
    - Will keep a list of at most 2*N length.
    - Trancate to latest N-msg, if more than 2*N


    Mode:
    - plaintext
    - asjsonstr
    
    Example:
    ````python
    class A(Errmsg):
        def __init__(self,arg_int,arg_str):
            Errmsg.__init__()
            if not isinstance(arg_int, int):
                self.errmsg_append(f"arg_int({arg_int}) is not int, but {type(arg_int)}.", title="Warn")
            if not isinstance(arg_str,str):
                self.errmsg_append(f"arg_str({arg_str}) is not str, but {type(arg_str)}.", title="Warn")
            
    a = A(1,2)
    a.errmsg_printall()
    a.errmsg_printlast(1)
    ````
    """
    titles = ["Info","Warning","Error"]
    def __init__(self,max_n_errmsg:int=0):
        # ,=arrmsg_data = [(msg_to_print, title)]
        self.errmsg_data=[] # type: List[Tuple[str,str]]
        self.max_n_errmsg = max_n_errmsg

    def errmsg_append(self,content:any,title="Info",mode="plaintext"):
        if mode == "plaintext":
            try:
                cont = str(content)
            except Exception as e:
                self.errmsg_printall()
                raise e
            
        elif mode == "asjsonstr":
            try:
                cont= json.dumps(content)
            except Exception as e:
                self.errmsg_printall()
                raise e
        else:
            raise Exception(f"Unknown mode: {mode}")
        
        self.errmsg_data.append((title,cont))
        self.errmsg_flush()



    def errmsg_flush(self):
        N =self.max_n_errmsg
        n = len(self.errmsg_data)
        if N>0 and n>=2*N:
            self.errmsg_data = self.errmsg_data[n-N:]
    
    def errmsg_clear(self):
        self.errmsg_data.clear()
        
    def errmsg_raise_if(self,alert_titles=["Error"], printall_before_raise = False):
        if not len(self.errmsg_data) ==0:
            cnt = {}
            for k in alert_titles:
                cnt[k] = 0
            for title,_ in self.errmsg_data:
                if title in alert_titles:
                    cnt[title]+=1
            
            if any(v > 0 for v in cnt.values()):
                if printall_before_raise:
                    self.errmsg_printall()
                raise Exception(f"Alerted message found: {', '.join(f'{k}: {v}' for k,v in cnt)}")
            
    def errmsg_count(self, titles = []):
        assert len(titles)>=1
        n = 0
        for title,_ in self.errmsg_data:
            if title in titles:
                n+=1
        return n
    
    def errmsg_fetch(self, titles = []):
        l = []
        for title,cont in self.errmsg_data:
            if title in titles:
                l.append((title,cont))
        return l
    
    def errmsg_drop(self, titles=[]):
        l = []
        ld = []
        for title,cont in self.errmsg_data:
            if title not in titles:
                l.append((title,cont))
            else:
                ld.append((title,cont))
        self. errmsg_data = l
        return ld
            
    
    def errmsg_msg2str(self,msg):
        """
        title,c = msg
        """
        title,c = msg
        msgstr = f"- {title}: {c}"
        return msgstr
    
    def errmsg_print(self,msg:Tuple):
        """
        msg = title:str,content:str
        """
        title,_ = msg
        _ms=  self.errmsg_msg2str
        if title == "Error":
            print_error(_ms(msg))
        else:
            print_notice(_ms(msg))
        
    
    def errmsg_printall(self,recursive=False,suffix=""):
        """ 
        recursive:
            Recursive on vars(self) if have `errmsg_printall`
        """
        
        if len(self.errmsg_data)>0:
            _ms=  self.errmsg_msg2str
            for msg in self.errmsg_data:
                self.errmsg_print(msg)
        else:
            print_notice("No errmsg.")

        if recursive == True:
            for name,m in vars(self).items():
                if hasattr(m,"errmsg_printall"):
                    print_notice(f"== Errmsg:{name}{suffix}")
                    m.errmsg_printall(recursive=recursive,suffix=f".{name}")
            
    def errmsg_printlast(self,n_last:int =1):
        if len(self.errmsg_data)>0:
            for i in range(n_last):
                self.errmsg_print(self.errmsg_data[-i])
        else:
            print_notice("No errmsg.")




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

import torch
from typing import Dict
class BufferDict(torch.nn.Module):
    def __init__(self,d:Dict,persistent:bool):
        super().__init__()
        for k,v in d.items():
            self.register_buffer(k,v,persistent=persistent)
        



# PLOT dict and keys


def is_valid_keys(keys,valid_keys:List[str],check_name:Optional[str]=None,raise_exception=True):

    if check_name is None:
        check_name ="is_valid_keys"
    invalid_keys = []

    for k in keys:
        if k not in valid_keys:
            invalid_keys.append(k)
    if raise_exception:
        assert len(invalid_keys)==0,f"{check_name}: {invalid_keys} not in {valid_keys}."
    else:
        return invalid_keys

def has_keys(required_keys:List[str],target_dict:dict,dict_name:Optional[str]=None,raise_exception =True):
    
    if dict_name is None:
        dict_name = 'has_keys'
    missing_keys = []
    for k in required_keys:
        if k not in target_dict:
            missing_keys.append(k)
    if raise_exception:
        assert len(missing_keys)==0,f"{dict_name}:{missing_keys} not found."
    else:
        return missing_keys



import collections
from typing import Callable,Iterable

import Levenshtein
def find_nearest_altkey(key,known_keys:Iterable):
    """
    Return altkey, dist

    dist ==0 , if key == altkey
    dist ==-1, if known_keys is empty
    """
    if key in known_keys:
        return key,0
    else:
        distl = []
        kl=[]
        i=-1
        for i,k in enumerate(known_keys):
            d = Levenshtein.distance(k,key)
            distl.append(d)
            kl.append(k)
        if i==-1:
            return None,-1 # known_keys is empty
        else:
            min_d = min(distl)
            j= distl.index(min_d)
            altkey = kl[j]
    return altkey,min_d

class MsgedDict(collections.UserDict):
    def __init__(self,*args,**kargs):
        super().__init__(*args,**kargs)
        self.msg_missingkey = None
        self.handler_missingkey = None
    def set_msg_missingkey(self,msg:str):
        """
        msg: string_template(key=key)
        """
        self.msg_missingkey = msg
    
    def set_handler_missingkey(self,f:Callable):
        """
        f(key:str,known_keys:Iterable)
        """
        self.handler_missingkey = f


    def __getitem__(self, key:str):
        m = self.msg_missingkey
        f = self.handler_missingkey
        if key not in self:
            if f is not None:
                msg=f(self,key)
            else:
                altkey,dist = find_nearest_altkey(key,self.keys())
                if dist ==-1:
                    msg = f"Key '{key}' not found. It's an empty dict."
                else:
                    msg = f"Key '{key}' not found. Do you mean '{altkey}' ?"
            raise KeyError(msg)
        else:
            return super().__getitem__(key)


def kwargs(*keywords):
    """
    NOTICE: Using @kwargs() 
    
    For dataclass to have keyword-arguments only restriction(Before Python 3.10)

    if len(keywords)==0 then ALL_KEYWORDS_ONLY

    https://stackoverflow.com/questions/49908182/how-to-make-keyword-only-fields-with-dataclasses

    Coming in Python 3.10, there's a new dataclasses.KW_ONLY sentinel that works like this:
    
    .. code-block:: python
    
        @dataclasses.dataclass
        class Example:
            a: int
            b: int
            _: dataclasses.KW_ONLY
            c: int
            d: int
    
    Any fields after the KW_ONLY pseudo-field are keyword-only.

    There's also a kw_only parameter to the dataclasses.dataclass decorator, which makes all fields keyword-only:

    .. code-block:: python
    
        @dataclasses.dataclass(kw_only=True)
        class Example:
            a: int
            b: int
    
    It's also possible to pass kw_only=True to dataclasses.field to mark individual fields as keyword-only.

    If keyword-only fields come after non-keyword-only fields (possible with inheritance, or by individually marking fields keyword-only), keyword-only fields will be reordered after other fields, specifically for the purpose of __init__. Other dataclass functionality will keep the declared order. This reordering is confusing and should probably be avoided."""
    
    def decorator(cls):
        @wraps(cls)
        def call(*args, **kwargs):
            sig = signature(cls)
            param_l = list(sig.parameters.keys())

            if len(keywords) == 0:
                kw_l = param_l
            elif any(kw not in param_l for kw in keywords):
                raise Exception(f"Decorator: Not all {keywords} in {cls.__name__}({param_l}).")
            else:
                kw_l = keywords

            for kw_in_need in kw_l:
                if kw_in_need not in kwargs and sig.parameters[kw_in_need].default != inspect.Signature.empty:

                    raise TypeError(f"{cls.__name__}.__init__() requires {kw_in_need} as keyword arguments")
            
            n_pos_or_kw = len(param_l)-len(kw_l)
            if len(args)>n_pos_or_kw:
                raise TypeError(f"{cls.__name__}.__init__() requires {kw_l} as keyword arguments")
            return cls(*args, **kwargs)
        
        return call

    return decorator

import re

def splitkeys(keystr:str,*,delim:str=" "):
    """
    space at beg/end will be ignored;
    delim at beg/end will be ignored;
    newline will be ignored;
    double-delim will be ignored;
    """
    try:
        ks = keystr
        assert len(delim) ==1,f"delim {delim}"
        ks = ks.replace("\n",delim)
        ks = re.sub(f"{delim}[ ]{{0,}}",f"{delim}",ks)
        ks = re.sub(f"[ ]{{0,}}{delim}",f"{delim}",ks)
        ks = re.sub(f"[{delim}]{{2,}}",f"{delim}",ks)
        ks = ks.lstrip().rstrip()
        ks = ks.lstrip(delim).rstrip(delim)
        keys = ks.split(delim)
    except Exception as e:
        print(f"== keystr = {keystr}")
        print(f"== `delim` = `{delim}`")
        import traceback
        print(traceback.format_exc())
        raise e
    return keys 


def _getitems(dictlike,keystr:str,*,delim:str=" "):
    """
    Prototype:
    - return tuple(keys), tuple(values)
    - `s[0]` or `s[text]` will turn to `s.0` or `s.text` as keys 

    You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.

    If you musr mix notations, re-factor your data structure! A sign of bad codes :-)


    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())
    
    See `splitkeys(keystr,delim)`
    
    If '[' in the first `k`, will catch a k2, and
    - take d[k][int(k2)], then
    - if failed, will take d[k][str(k2)].

    Notice that the k2-logic(int or str) MUST be the same for all keys.

    For dataclasses, use `var(dataclass_obj)` as `dictlike`
 

    """
    keys = splitkeys(keystr,delim=delim)
    assert len(keys)>=1,f"{keys}"
    if "[" in keys[0]:
        # check k2-logic
        k,k2 = keys[0].split("[")
        k2= k2[:-1]
        try:
            int(k2)
            isint=True
        except ValueError as e:
            str(k2)
            isint=False

        kp = tuple(key.split("[") for key in keys)
        kp = tuple( (k,k2[:-1]) for k,k2 in kp)

        
        
        try:
            if isint:
                tp= tuple(dictlike[k][int(k2)] for k,k2 in kp)
            else:
                tp= tuple(dictlike[k][str(k2)] for k,k2 in kp)
        except Exception as e:
            print(f"== kp={kp}")
            import traceback
            print(traceback.format_exc())
            raise e

        return tuple(f"{k}.{k2}" for (k,k2) in kp), tp
    else:
        return keys, tuple(dictlike[k] for k in keys)

from typing import Dict
def getitems(dictlike,keystr:str,*,delim:str=" ")->tuple:
    """
    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())

     You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.
    

    Example:

    ```python
    getitems({"a":0,"b":1},"  a b ") # (0,1)

    getitems(var(object),"names[0], tags[0]",delim=",") # (obj.names[0], obj.tags[0])
    ```

    See `splitkeys(keystr,delim)` and `_getitems` for more.
    

    """
    return _getitems(dictlike,keystr,delim=delim)[1]


def getitems_as_dict(dictlike,keystr:str,*,delim:str=" "):
    """
    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())

     You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.
    

    Example:

    ```python
    getitems({"a":0,"b":1},"  a b ") # {"a":0,"b":1}

    getitems(var(object),"names[0], tags[0]",delim=",") # ("names.0": obj.names[0], "tags.0": obj.tags[0])
    ```

    See `splitkeys(keystr,delim)` and `_getitems` for more.

    """
    try:
        kl,vl = _getitems(dictlike,keystr,delim=delim)
    except KeyError as e:
        key = e.args[0]
        print(f"- Error: the key `{key}` not found in dictlike.")
        raise e
    return {k:v for k,v in zip(kl,vl)}

def updateattrs(obj,newdict):
    for k,v in newdict.items():
        setattr(obj,k,v)

def dargs_for_calling(f:Callable,d:dict):
    """Inspect the arg list of Callable f, fetch items for d and return;
    """
    import inspect
    args = inspect.getfullargspec(f)
    fal = [n for n in args.args]+[n for n in args.kwonlyargs]
    fd=  {}
    for k in fal:
        if k in d:
            fd[k] = d[k]
    
    return fd

def call_by_inspect(f:Callable,d:dict,**kwargs):
    """
    dargs = dargs_for_calling(f,d))

    dargs.update(kwargs)

    Return f(**d)
    """
    dargs = dargs_for_calling(f,d)
    dargs.update(kwargs)
    return f(**dargs)

def as_int(s):
    """
    Return None if failed;
    Or, return integer;

    Notice 0(int) is false-if-condition;
    """
    try:
        i = int(s)
    except:
        return None
    else:
        return i
    


def dargs_for_formatting(string_template:str,d:dict):
    """
    Return dict

    ONLY support keyword-format: {a};
    Raise exception if {}, {{}}, {0}
    """
    p = r"({(?:[^{}]*|(?R))*})" # will capture the brackets and contents between;
    s=string_template

    pt = regex.compile(p)
    m = regex.findall(pt,s)
    args = []
    for t in m:
        t2=  t[1:-1]
        if "{" in t2:
            raise Exception(f"Not supported: recursive: {t}")
        if as_int(t2) is not None:
            raise Exception(f"Not supported: positional: {t}")
        args.append(t2)
    dargs = {n:d[n] for n in args if n in d}
    return dargs

def format_by_re(string_template:str,d:dict,**kwargs):
    """Only kwargs-string-format supported.
    """

    dargs = dargs_for_formatting(string_template,d)
    dargs.update(kwargs)
    s2=  string_template.format(**dargs)
    return s2


# PLOT config
import json
from pathlib import PosixPath
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
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)






# ==== From  /home/tjrym/workspace/MyHumbleADMM7/RenNet/framework/Core/RyLog/__init__.py, 2023-05-17

import yaml
from pprint import pformat
class Hparams:
    """
    save as yaml at

    <save_dir>/<hparams_stem_prefix>_version_<version>.yaml


    Generally ckp is in :
    lightning_logs/version_{version number}/epoch_{epoch number}-step_{global_step}.ckpt

    and to access them:

    version_number -> trainer.logger.version
    epoch_number -> trainer.current_epoch
    global_step -> trainer.global_step
    """
    hparams_stem_prefix = f"my_hparams"
    def __init__(self,save_dir,version,):
        """
        Delay the reference of self.version until .save
        """
        self.version = version # if None, then must be set before .save
        self.save_dir= save_dir
        self.data = {
            "args":[],
            "kargs":{}
        }


    @property
    def args(self)->list:
        """
        Return a read/write list
        """
        return self.data["args"]
    
    @args.setter
    def args(self,l):
        self.data["args"]= l

    @property
    def kargs(self)->dict:
        """
        Return a read/write dict
        """
        return self.data["kargs"]
    
    @kargs.setter
    def kargs(self,d):
        self.data["kargs"]=  d

    def extend_args(self,*args):
        self.args.extend(args)

    def update_kargs(self,**kargs):
        self.kargs.update(kargs)

    @property
    def fdir(self):
        return Path(self.save_dir).as_posix()

    @property
    def fp(self):
        return Path(self.fdir,f"{self.hparams_stem_prefix}_version_{self.version}.yaml").as_posix()# Avoid pl.trainer find version folder occupied.

    def save(self):
        """
        Return file path.
        """
        assert self.version is not None, f"self.version {self.version}"
        Path(self.fdir).mkdir(exist_ok=True,parents=True)
        fp = self.fp
        if Path(fp).exists():
            os.remove(fp)
        with open(fp,"x") as f:
            sss = yaml.safe_dump(self.data)
            f.write(sss)
        return Path(fp).as_posix()

    def load(self):
        """
        Return a dict or sth?
        """
        fp = self.fp
        with open(fp,"r") as f:
            sss = f.read()
            d = yaml.safe_load(sss)
        return d



import re
def collect_globals(global_dict:dict,type_to_match=[float,int,str,dict,list],ignored_keys=[],*,ignored_keys_regex=False,as_str=False):
    """
    Dict = collect_globals(globals())

    Skip `__xx__`
    Skip `_xx`
    """
    d ={}
    for key,obj in global_dict.items():
        if re.match("__.*__",key) is not None:
            continue
        if re.match("_.*",key) is not None:
            continue
        if ignored_keys_regex:
            raise NotImplementedError()
        if key in ignored_keys:
            continue
        for cls in type_to_match:
            if isinstance(obj,cls):
                d[key] = obj

    if as_str :
        s = pformat(d)
        lines = s.split('\\n')
        def count_indent_space(s):
            """count space, not \\t.
            """
            n = 0
            for ch in s:
                if ch == ' ':
                    n +=1
            return n
        n_last = 0
        new_lines = []
        for seg in lines:
            if seg !='':
                r = seg.split('\n')
                if len(r)==1:
                    n = count_indent_space(r[0])
                elif len(r)==2 and r[1] =='':
                    n = count_indent_space(r[0])
                else:
                    n = count_indent_space(r[-1])
                n_last = n
            else:
                n = n_last
            new_lines.append((''*n) +seg)
        s = '\n'.join(new_lines)
        return s
    else:
        return d




# ==== yaml dumper/loader
import torch

class MyDataclass:
    pass


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


from typing import Type,Any,TypeVar 
T = TypeVar("T")

def add_representers(target_dumper_type:Type[yaml.SafeDumper], representers:List[Callable[[yaml.SafeDumper,TypeVar("T"),str],yaml.SafeDumper]], prefix:str="represent"):
    """
    - The function(i.e. representer) is Callable(dumper:yaml.SafeDumper, data:type_hint, tag:str)
    - ONLY take functions that startwith(f"{prefix}_") if prefix is not None;
    - Remove the f"{prefix}_";
    - Make tag `!PosixPath` from `posix_path`;
    - target_dumper_type.add_representer(T, lambda data: func(data, tag))

    """
    representers = filter(lambda f:isinstance(f,Callable),representers)
    if prefix is not None:
        names = []
        _funcs = []
        for func in representers:
            if func.__name__.startswith(f"{prefix}_"):
                names.append(func.__name__[len(f"{prefix}_"):])
                _funcs.append(func)
        representers = _funcs
    else:
        names= [func.__name__ for func in representers]
            
    for func,func_name in zip(representers,names):
        arg_spec = inspect.getfullargspec(func)
        type_hint = arg_spec.annotations.get(arg_spec.args[1]) # func(self,data:type_hint)
        tag = "!"
        tag += "".join(s.capitalize() for s in func_name.split("_"))
        target_dumper_type.add_representer(type_hint, lambda dumper,data: func(dumper, data, tag))



def add_constructors(target_loader_type:Type[yaml.SafeLoader], constructors:List[Callable[[yaml.SafeLoader,TypeVar("node"),str],yaml.SafeLoader]], prefix:str="constructor"):
    """
    - The function(i.e. constructor) is Callable(dumper:yaml.SafeLoader, node, tag:str)
    - ONLY take the functions with the startwith(f"{prefix}_");
    - Remove f"{prefix}_"
    - Make tag `!PosixPath` from `posix_path`;
    - target_dumper_type.add_representer(T, lambda data: func(data, tag))

    """
    constructors = filter(lambda f:isinstance(f,Callable),constructors)
    if prefix is not None:
        names = []
        _funcs = []
        for func in constructors:
            if func.__name__.startswith(f"{prefix}_"):
                names.append(func.__name__[len(f"{prefix}_"):])
                _funcs.append(func)
        constructors = _funcs
    else:
        names= [func.__name__ for func in constructors]

    for func in constructors:
        func_name = func.__name__
        if prefix is not None:
            assert func_name.startswith(f"{prefix}_")
            func_name = func_name[len(f"{prefix}_"):]
        tag = "!"
        tag += "".join(s.capitalize() for s in func_name.split("_"))
        target_loader_type.add_constructor(tag, lambda loader,node: func(loader, node, tag))



def merge_dumpers(target:yaml.SafeDumper, from_dumpers:List[yaml.SafeDumper]):
    for dumper in from_dumpers:
        for representer_type, representer_method in dumper.representers.items():
            # Check if the tag has already been registered
            if representer_type in target.representers.keys():
                raise ValueError(f"Tag '{representer_type}' already has a representer registered")
            target.add_representer(representer_type, representer_method)
    return target


class RenNetDumper(yaml.SafeDumper):
    def represent_posix_path(self, data:PosixPath, tag:str):
        return self.represent_scalar(tag, str(data))
    
    def represent_small_tensor(self,data:torch.Tensor, tag:str):
        data = data.detach().cpu()
        assert data.numel()<16,f"Only small tensor(numel<16) supported."
        return self.represent_mapping(tag,{
            "dtype":str(data.dtype),
            "data":totuple(totuple(data)),
        })

add_representers(RenNetDumper,list(vars(RenNetDumper).values()),prefix="represent")

class RenNetLoader(yaml.SafeLoader):
    def construct_posix_path(self, node, tag:str):
        return Path(node.value)
    def construct_small_tensor(self,node, tag:str):
        d= self.construct_mapping(node, deep=True)
        tsr = torch.tensor(d["data"],dtype=d["dtype"])
        return tsr

add_constructors(RenNetLoader,list(vars(RenNetLoader).values()), prefix="constructor")