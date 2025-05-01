#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import scipy
def interp_spd(wl_old, spd_func, wl_new):
    '''
    - spd_func shapes [wavelength, channels]
    '''
    
    if len(spd_func.shape)==1:
        spd_func=spd_func[:,np.newaxis]
        
    spd_new = []
    for ch_ in range(spd_func.shape[1]):
        f_interp1d = scipy.interpolate.interp1d(
            wl_old.squeeze(), spd_func[:, ch_], fill_value='extrapolate')
        spd_ch = f_interp1d(np.array(wl_new))
        spd_new.append(spd_ch)

    spd_new = np.array(spd_new).T

    return spd_new

