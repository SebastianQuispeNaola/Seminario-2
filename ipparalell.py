# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:17:18 2021

@author: Intel
"""

import os
import ipyparallel as ipp

cluster = ipp.Cluster(n=4)
with cluster as rc:
    ar = rc[:].apply_async(os.getpid)
    pid_map = ar.get_dict()