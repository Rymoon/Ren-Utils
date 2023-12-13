

from functools import wraps
import numpy as np
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
    
    Return ndarray 
    
    gaussian_noise() return a float
    """
    
    if shape is not None:
        n = 1
        for v in shape:
            n = n*v
        noise = np.random.rand(*shape)
    else:
        n = 1
        noise = np.random.rand()
    
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
        rt.state("state_for_randomness_check__i",i+gaussian_noise())
    
    
    
    
    