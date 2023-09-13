import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from warnings import warn

from ..rennet import (call_by_inspect, getitems_as_dict,
                                        root_Results)


# ==== utils ====
def do_remap(v):
    return (v-v.min())/(v.max()-v.min())
def list_pictures(directory, ext:Tuple|str=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif','tiff','gif')):
    """Lists all pictures in a directory, including all subdirectories.

    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures

    # Returns
        a list of paths


    # Copy from keras_preprocessing.image.utils::list_pictures
    """
    assert Path(directory).exists(), f"Not exists: {directory}"
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]
    


from typing import List
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import overload
from ren_utils.rennet import Errmsg

class ImageDataset(Dataset,Errmsg):
    """
    To subclass, please implement:
    + an __init__(); see ImageDataset.limit for reason
    
    * If layers is None, then it's a placeholder dataset, wont be initialized, i.e., call self.init(...);
    * else, layers[0] must be a valid List[str] in any case; Others can be [] or None.
    
    """
    @overload
    def __init__(self):
        """Create an empty object."""
        pass
    @overload
    def __init__(self, layers:List[List[str|Path]]|Any, resize_size: None|Tuple[int,int]=None):
        """
        By default, 
        1. self._prepare_datapair(**layers) to create matched data-pair;
        2. _transform(layers,i,j) take:
        
            layers = [layer0,layer1,...]
            layern = [path0,path1,...]
        
        Override these to change behaviour.
        [# FUTURE] Create a h5 version at first run;
        """
    def __init__(self, layers:List[List[str|Path]]=None, resize_size: None|Tuple[int,int]=None):
        Dataset.__init__(self)
        Errmsg.__init__(self)
        # List of files
        # self._files = [p for p in Path(folder).glob(f'**/*.jpg')]
        self._length = 0
        if layers is not None:
            try:
                errcode = 0
                layers[0]
                
                errcode = 1
                len(layers[0])
                
                errcode = 2
                assert len(layers[0])>=1
                
                errcode = 3
                layers[0][0]
            except Exception as e:
                # layers[0] should be valid list
                print(f"* ImageDataset.__init__::errcode={errcode}")
                raise e
            
            self.init(layers, resize_size)
    
    def init(self,layers,resize_size):
        """
        1. _prepare_datapair
        2. transform
        3. length
        """
        # Override paired_layers if not the case.
        self.paired_layers = self._prepare_datapair(layers) # type: List[List[str|Path]]
        
        

        # Default transformas: 
        #   1. Image.open(path) 
        #   2. torchvision:
        #       - resize 
        #       - convert to tensor, i.e.,  PIL image [0,255] to float tensor[0,1]
        # Called at __getitem__(index)
        
        # transform (default)
        self._trans = [
            torchvision.transforms.Compose(
            ([torchvision.transforms.Resize(resize_size)] if resize_size is not None else [])
            +[torchvision.transforms.ToTensor(),]
            ),
        ]
        
        
        self._length = len(self.paired_layers[0])
        
    def transform(self, layers:List[List[str]],i,j):
        """
        Default transformas: 
          1. Image.open(path) 
          2. torchvision: (self._trans)
              - resize 
              - convert to tensor, i.e.,  PIL image [0,255] to float tensor[0,1]
        Called at __getitem__(index)
        """
        if layers[i] is None:
            return None
        else:
            p = layers[i][j] # str(path)
            p = Path(p)
            assert p.exists(), p.as_posix()
            o = Image.open(p)
            for _t in self._trans:
                o = _t(o)
            return o
        

 
    def _prepare_datapair(self,layers):
        """
        1. Loop: Follow the order of layers[0],
        2. Match: Search in other layers[i], for filename of the same stem.
            * Achieve by create the {stem:path}.
            * put None if not found.
        
        For example, if  
        `self.layers        = [[...90 pathes], None, [...100 pathes]]`
        then 
        `self.paired_layers = [[...90 pathes], None, [...90 pathes]]`
        

        """
        
        assert len(layers[0])>0
        _layers_i=  []
        for i in range(_layers_n):
            if layers[i] is not None:
                _layers_i.append(i) 
        
        _layers_n = len(layers)
        
        # layers[i] != layers[j] is ok
        for i in _layers_i:
            if len(layers[0]) == len(layers[i]):
                msg = f"length of layers[.] not equal, 0:{len(layers[0])}, {i}:{len(layers[i])}"
                self.errmsg_append(msg,"_prepare_datapair - length")
    
        
        stem_ld = [None]*_layers_n
        for i in _layers_i:
            stem_ld[i]=  {Path(p).stem:p for p in layers[i]}
        
        paired_layers = [None]*_layers_n
        for i in _layers_i:
            paired_layers[i] = []
        
        for j in range(len(layers[0])):
            stem = Path(layers[0][j]).stem # order of layers[0]
            for i in _layers_i:
                if stem in stem_ld[i]:
                    paired_layers[i].append(stem_ld[i][stem]) 
                else:
                    paired_layers[i].append(None)
                    self.errmsg_append(("Missing",(i,j,stem)),"_prepare_datapair - pair") 
        
        self.errmsg_printall()
        self.errmsg_raise_if(["_prepare_datapair - pair"])
        return paired_layers

    def __len__(self):
        """
        Size of the dataset
        """
        return self._length

    def __getitem__(self, index: int):
        """
        Return item, a list;
        
        [# FUTURE] HDF5 file
        
        To get item[i],
        - Try get p=layers[i][index];
        - Iteratively apply func(p) in the list, _transform[i];
        """
        item = []
        
        j = index
        for i in range(len(self.paired_layers)):
            out = self.transform(self.paired_layers,i,j)             
            item.append(out)
        return item
    

    def limit(self,n_limit:int|None):
        o= type(self)()
        o.transform = self.transform
        o.paired_layers = self.paired_layers
        if n_limit is None:
            o._length = len(self.paired_layers[0])
        else:
            assert n_limit>=0
            o._length = min(len(self.paired_layers[0]), n_limit)

        return o
