import json
import os
import re
import shutil
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import torch
import torch.nn as tnn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ==== From MyHumbleADMM7/RenNet/framework/Core/RyOpr/__init__.py , 2023-05-16
import math
import torch
import torch.nn.functional as torchf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import torch.nn.functional as tnnf


# Util
TORCH_VERSION = tuple()
for v in torch.__version__.split("."):
    try:
        v = int(v)
    except:
        pass
    else:
        TORCH_VERSION +=(v,)
        
    
# PLOT Complex
def cplx_mul_tc(a:torch.Tensor,b:torch.Tensor):
    real = a[...,0]*b[...,0]-a[...,1]*b[...,1]
    real= real.unsqueeze_(-1)
    imag = a[...,0]*b[...,1]+a[...,1]*b[...,0]
    imag= imag.unsqueeze_(-1)

    return torch.cat((real,imag),-1)
def cplx_div_tc(a:torch.Tensor,b:torch.Tensor):
    ar,ai=a[...,0],a[...,1]
    br,bi=b[...,0],b[...,1]
    magb = br*br+bi*bi
    real = (ar*br/magb+ai*bi/magb).unsqueeze_(-1)
    imag = (ai*br/magb-ar*bi/magb).unsqueeze_(-1)
    res= torch.cat((real,imag),-1)
    return res
def cplx_abs(cplx):
    real = cplx[...,0]
    imag = cplx[...,1]
    return (real**2+imag**2).sqrt()
cmul =      cplx_mul_tc
icmul =     cplx_div_tc
cabs =      cplx_abs
# PLOT fft

if TORCH_VERSION<(1,8):
    rfft2_tc = lambda X:torch.rfft(X,2,onesided=False)
    irfft2_tc = lambda X:torch.irfft(X,2,onesided = False)
else:
    def rfft2_tc(X:torch.Tensor):
        """real to old_complex
        """
        cplx = torch.fft.fft2(X, dim=(-2, -1))
        t2 = torch.stack((cplx.real, cplx.imag), -1)
        return t2
    
    def irfft2_tc(X:torch.Tensor):
        """old_complex to real.
        """
        cplx = torch.fft.ifft2(torch.complex(X[..., 0], X[..., 1]), dim=(-2, -1)) 
        tr = cplx.real
        return tr
def psf2otf_tc(kernel:torch.Tensor,out_shape:Tuple[int,int]):
    ''' for each 1x...x1x(height,width) do psf2otf
    outshape: (int,int)
    '''
    out_shape = out_shape[-2],out_shape[-1]
    psf = kernel
    psf_shape = (psf.shape[-2],psf.shape[-1])
    pad_shape = (out_shape[0] - psf_shape[0],out_shape[1] - psf_shape[1])
    # F.pad dim-order : 3-2-1-0   pre,post
    psf = torchf.pad(psf,(0,pad_shape[1],0,pad_shape[0]))
    roll_shape = (-math.floor(psf_shape[0]/2),-math.floor(psf_shape[1]/2))
    psf = torch.roll(psf,roll_shape,(-2,-1))
    res= rfft2_tc(psf)
    return res

_cr  =  lambda K:(round((K.shape[-2]-1)/2),round((K.shape[-1]-1)/2))
psf2otf =     psf2otf_tc

def deconv2fft(Du:torch.Tensor,D:torch.Tensor):
    '''
    
    ffted --> deconv2 --> ffted
    
    - D: C, 1, H, W, 2
    - Du:B, C, H, W, 2 

    - u: B, C, H, W, 2
    '''
    
    d = D.sum(dim=1)
    d = d.unsqueeze(0) # 1, C, H, W, 2
    u = icmul(Du,d)     # B, C, H, W, 2
    return u
def fft2(a):
    try:
        v = rfft2_tc(a)
        return v
    except Exception as e:
        print('==== DEBUG ====')
        print(a.shape)
        print('==== ===== ====')
        raise e
ifft2  =   irfft2_tc
    
# PLOT  Convolution; Manage channel-dim after convolution
def conv2c_parallel(D,u):
    """ 
    - D: oC,1,kH,kW

    - u: B,oC,H, W

    - Du:B,oC,H, W
    
    
    padu = tnnf.pad(u,pad=(pW,pW,pH,pH),mode = "circular")
    
    Du = tnnf.conv2d(padu, D, groups = oC, bias=None, padding="valid")
    """
    assert D.device ==u.device,f"{D.device},{u.device}"
    oC,iC1,kH,kW = D.shape
    B,oC2,iH,iW = u.shape
    assert kH%2==1,f"{kH}"
    assert kW%2==1,f"{kW}"

    if oC !=oC2:
        raise Exception("D={},u={}".format(D.shape,u.shape))
        
    D = D.flip((-2,-1))# NOTICE to simulate numpy's convolution2d
    
    pH,pW= (kH-1)//2, (kW-1)//2

    padu = tnnf.pad(u,pad=(pW,pW,pH,pH),mode = "circular")
    Du = tnnf.conv2d(padu, D, groups = oC, bias=None, padding="valid")
    
    return Du
    

def squeezek(D,R=None):
    rH,rW = _cr(D)
    def _eqz(a):
        return abs(a)<1e-8
    with torch.no_grad() :
        del_rH = 0
        for i in range(rH):
            if _eqz(D[...,i,:].sum()) and _eqz(D[...,-i,:].sum()):
                del_rH+=1
            else:
                break
    if del_rH==rH:
        D = D[...,rH:rH+1,:]
    elif del_rH==0:
        pass
    else:
        D= D[...,del_rH:-del_rH,:]
        
    with torch.no_grad() :      
        del_rW = 0
        for i in range(rW):
            if _eqz(D[...,:,i].sum()) and _eqz(D[...,:,-i].sum()):
                del_rW+=1
            else:
                break
    if del_rW == rW:
        D = D[...,:,rW:rW+1]
    elif del_rW ==0:
        pass
    else:
        D = D[...,:,del_rW:-del_rW]

    if R is not None:
        h,w = D.shape[-2],D.shape[-1]
        assert 2*R+1>=h and 2*R+1>=w,f"{R},{D.shape}"
        assert h%2==1 and w%2==1,f"{R},{D.shape}"
        if 2*R+1>h or 2*R+1>w:
            from torch.nn.functional import pad
            h_pad = (2*R+1-h)//2
            w_pad = (2*R+1-w)//2
            D = pad(D,(w_pad,w_pad,h_pad,h_pad),'constant',0)
    return D
        

def conv2k(D1,D2):
    '''
    D_1^T D_2
    Input:
        - D1 of oC1,iC1=1,kH1,kW1
        - D2 of oC2,iC2=1,kH2,kW2  
        
        oC1==oC2
        
    Output:
        D of oC2, iC1=1, kH, kW


    Assertion ItI == I
    '''
    
    oC1,iC1,_,_ = D1.shape
    oC2,iC2,_,_ = D2.shape
    if oC1!=oC2:
        raise Exception(oC1,oC2)
        
    if iC1!=iC2 or iC1!=1:
        raise Exception(iC1,iC2)
        
    oC = oC1
        
    *_,rH,rW = D1.shape
    *_,iH,iW = D2.shape
    
    
    D2 =D2.reshape(1,oC,iH,iW)
    DtD = tnnf.conv2d(D2,D1,groups =oC,bias=None, padding=(rH,rW))
    DtD = DtD.reshape(oC,1,DtD.shape[-2],DtD.shape[-1])

    
    return DtD

def CKtoC(x:torch.Tensor,*,C,dim):
    """
    dim of C*K
    """
    sp = tuple(x.shape)
    K = sp[dim]//C
    sp2 = sp[:dim] + (C, K) +sp[dim+1:]
    x2= x.reshape(sp2)
    y = x2.sum(dim=dim+1)
    return y

def CtoCK(x:torch.Tensor,*,K,dim):
    """
    dim of C
    """
    sp = tuple(x.shape)
    C=  sp[dim]
    sp2 = sp[:dim+1] + (K,)+ sp[dim+1:]
    x2= x.unsqueeze(dim+1).expand(sp2)
    sp3 = sp[:dim] + (C*K,)+ sp[dim+1:]
    y = x2.reshape(sp3)
    return y

def KtoCK(x:torch.Tensor,*,C,dim):
    """
    dim of K
    """
    sp = tuple(x.shape)
    K = x.shape[dim]
    sp2 = sp[:dim] + (C,) + sp[dim:]
    x2 = x.unsqueeze(dim=dim).expand(sp2)
    sp3 = sp[:dim] + (C*K,)+ sp[dim+1:]
    y = x2.reshape(sp3)
    return y

def get_IK(Ker,*,C,r:float,do_squeeze=False):
    """
    Ker: C*K,1,kH,kW

    ker = CKtoC(Ker,C,dim=0)

    get I+ r*ktk
    """
    ker = CKtoC(Ker,C=C,dim=0)
    ktk = conv2k(ker,ker)
    if do_squeeze:
        ktk = squeezek(ktk)
    new_size = (ktk.shape[-1])
    new_R = (new_size-1)//2
    I = torch.zeros_like(ktk)
    I[:,:,new_R,new_R] = 1

    D = I + r*ktk
    return D

# PLOT init kernels
def getKernel_dispatch(init_args:str,n,R):
    """Arrays in list returned may share memory.

    init_args = init_type[#modification]
    
    .. warning::

        No gurantee of sharing memory of not, for kernels in the list. Do clone or detach by yourself.
    
    Return::
    
        List[Ker], len n,    
        
        where Ker: 1,1,kH,kW
    """
    modi = None
    if '#' in init_args:
        init_type,modi = init_args.split('#')
    else:
        init_type = init_args

    if init_type =='DxDy':
        assert n>=2
        m = n//2
        Dx,Dy = getKernel_DxDy(R)

        D = [Dx,Dy]*m
        if n-2*m>0:
            D.extend([Dx,Dy][:n-2*m])
        result  =D
    else:
        raise NotImplementedError(f"{init_type}")
    if modi is not None:
        if re.match("plus_eps([\.0-9]*)",modi) is not None:
            o = re.match("plus_eps([\.0-9]*)",modi)
            eps = o.group(1)
            eps = float(eps)
            result = [r + eps for r in result]
        else:
            raise NotImplementedError(f"{modi}")
    return result

DTYPE = torch.float32
zeros= lambda *shape,**kargs:torch.zeros(*shape,**kargs,dtype = DTYPE)
ones= lambda *shape,**kargs:torch.ones(*shape,**kargs,dtype = DTYPE)
rand= lambda *shape,**kargs:torch.rand(*shape,**kargs,dtype = DTYPE)

def getKernel_Dx(R):
        Dx = zeros(1,1,2*R+1,2*R+1)
        Dx[...,R,R-1]=1
        Dx[...,R,R]=-1
        return Dx

def getKernel_Dy(R):
        Dy = zeros(1,1,2*R+1,2*R+1)        
        Dy[...,R-1,R]=1
        Dy[...,R,R]=-1  
        return Dy

def getKernel_DxDy(R):
        Dx = getKernel_Dx(R)
        Dy = getKernel_Dy(R)
        return Dx,Dy

# PLOT misc
def softt(x:torch.Tensor,a:float):
    "Soft-thresholding"
    relu = tnn.functional.relu
    y=  relu(x-a)-relu(-x-a)
    return y

