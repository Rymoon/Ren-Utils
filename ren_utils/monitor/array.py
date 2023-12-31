

from functools import wraps
import numpy as np
import torch
func_under_CWC = {}
def count_when_calling(counter_d:dict={"called":0}):
    def deco(f):
        func_under_CWC[f.__name__] = {"counter_d":counter_d}
        @wraps(f)
        def g(*args,**kargs):
            return f(*args,**kargs,CWC_counter_d = counter_d)
        return g
    return deco

@count_when_calling({"n_float_sampled":0})
def gaussian_noise(shape=None,*,CWC_counter_d):
    """
    N(0,1)
    
    Return tensor
    
    gaussian_noise() return a float
    """
    
    if shape is not None:
        n = 1
        for v in shape:
            n = n*v
        noise = torch.randn(shape)
    else:
        n = 1
        noise = torch.randn(1).cpu().item()
    
    CWC_counter_d["n_float_sampled"]+=n
    return noise

def get_nfloats_sampled_gaussian_noise():
    return func_under_CWC[gaussian_noise.__name__]["counter_d"]["n_float_sampled"]

from ren_utils.monitor import RunTracer
def state_for_randomness_check(rt: RunTracer):
    """For numpy random engine
    Generate 5 float guassian and state;
    
    Call gaussion_noise();
    
    key = state_for_randomness_check__x
    x={0,1,2,3,4}
    """
    for i in range(5):
        rt.state(f"state_for_randomness_check__{i}",i+gaussian_noise())
    
    
    
import numpy as np
import torch
def save_rng_state(key,rt:RunTracer):
    rt.state(f"{key}__torch-random-state",torch.get_rng_state())
    rt.state(f"{key}__numpy-random-state",np.random.get_state())
    
def load_rng_state(key,rt:RunTracer):
    torch.set_rng_state(rt.load_state(f"{key}__torch-random-state"))
    np.random.set_state(rt.load_state(f"{key}__numpy-random-state"))
    
    
    
def RGB2BGR(a,i:int):
    """
    RGB2BGR on dim-i
    """
    if isinstance(a,torch.Tensor):
        a_ = torch.swapaxes(a,0,i)
        a_ = torch.stack([a_[2,...],a_[1,...],a_[0,...]],dim=0)
        a = torch.swapaxes(a_,0,i)
    elif isinstance(a,np.ndarray):
        a_ = np.swapaxes(a,0,i)
        a_ = np.stack((a_[2,...],a_[1,...],a_[0,...]),axis=0)
        a = np.swapaxes(a_,0,i)
    else:
        raise Exception(f"NotSupportedType {type(a)}")
    return a
    
def BGR2RGB(a,i):
    return RGB2BGR(a,i)


def PosImg2RealImg(a):
    """
    [-1,1] to [0,1]
    """
    return (a*2)-1

def RealImg2PosImg(a):
    """
    [0,1] to [-1,1]
    """
    return (a+1)/2