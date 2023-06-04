# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:14:03 2020

@author: jy-3
"""

import numpy as np
import torch 
a = np.array([2,2,2])
b = torch.from_numpy(a)
print(a)

c = torch.randint(1,10,[3,3])
print(c)