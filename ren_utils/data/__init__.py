import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

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

class ImageDataset(Dataset):
    """
    For subclass, please implement an __init__(); see ImageDataset.limit for why

    If no masks/samples/labels, None is ok
    """
    @overload
    def __init__(self):
        """Create an empty object."""
        pass
    @overload
    def __init__(self, layers:List[List[str|Path]], resize_size: None|Tuple[int,int]=None):
        """
        layers = [[p_imgs layer-0],[p_imgs layer-1],[...],...]
        
        Use self.prepare_datapair(**layers) to create matched data-pair;

        _non_init: create a empty, non-initialized object;
        
        [# FUTURE] Create a h5 version at first run;
        """
    def __init__(self, layers:List[List[str|Path]]=None, resize_size: None|Tuple[int,int]=None):
        super().__init__()
        # List of files
        # self._files = [p for p in Path(folder).glob(f'**/*.jpg')]
        if layers is not None :
            self.init(layers, resize_size)
        else:
            self._length = 0
    
    def init(self,layers,resize_size):
        self.paired_layers = self.prepare_datapair(layers) # type: List[List[str|Path]]
        

        # Transformations to resize the image and convert to tensor
        # Remap values to [0,1],
        self._transform = torchvision.transforms.Compose(
            ([torchvision.transforms.Resize(resize_size)] if resize_size is not None else [])
            +[torchvision.transforms.ToTensor(),]
        )

        self._length = len(self.paired_layers[0])

    
    def size_raw(self)->Tuple[int,int]:
        raise NotImplementedError()
        
    def prepare_datapair(self,layers):
        """
        A list of list[pathes], each layer are properly sorted,so that
        the ([id][0],[id][1],...) is a data-pair;
        
        Match by stem, following layers[0];
        
        Return [[layer-0:path-to-image,...],[layer-1 pathes],...]
        """
        
        assert len(layers)>0
        assert layers[0] is not None
        assert len(layers[0])>0
        for i in range(len(layers)-1):
            if layers[i+1] is not None:
                assert len(layers[0]) == len(layers[i+1]), f"0:{len(layers[0])}, {i+1}:{len(layers[i+1])}"
    
        
        stem_ld = []
        for l in layers:
            if l is not None:
                stem_ld.append({Path(p).stem:p for p in l} )
            else:
                stem_ld.append(None)
        
        paired_layers = [[] for _ in range(len(layers))]
        for i in range(len(layers[0])):
            stem = Path(layers[0][i]).stem
            for il in range(len(layers)):
                if stem_ld[il] is not None:
                    paired_layers[il].append(stem_ld[il][stem]) 
                else:
                    paired_layers[il].append(None) 
        return paired_layers

    def __len__(self):
        """
        Size of the dataset
        """
        return self._length

    def __getitem__(self, index: int):
        """
        Get an image

        If layers[i] is None, then paired_layers[i] is None.
        If paired_layers[il] is None, then vp[il] is None. No image opened.
        if vp[il] is None, then no self._transform applied. as None. 

        """
        vp = tuple()
        for il in range(len(self.paired_layers)):
            if self.paired_layers[il] is None:
                img = None
            else:
                if self.paired_layers[il][index] is None:
                    img = None
                else:
                    try:
                        img = Image.open(self.paired_layers[il][index]) if self.paired_layers[il] is not None else None
                    except Exception as e:
                        print(f" - il={il}")
                        print(f" - len(self.paired_layers[il])={len(self.paired_layers[il])}")
                        print(f" - self.paired_layers[il][:20]={self.paired_layers[il][:20]}")
                        print(f" - self.paired_layers[il][index]={self.paired_layers[il][index]}")
                        raise e
            vp+=(img,) 
        
        transformed =  tuple(self._transform(v) if v is not None else None for v in vp)
        return transformed
    

    def limit(self,n_limit:int|None):
        o= type(self)()
        o._transform = self._transform
        o.paired_layers = self.paired_layers
        if n_limit is None:
            o._length = len(self.paired_layers[0])
        else:
            assert n_limit>=0
            o._length = min(len(self.paired_layers[0]), n_limit)

        return o
