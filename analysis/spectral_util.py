import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
import os 
import pathlib
import re
import PIL
import cv2

from PIL import Image


def listup_nuance_spcube_files(srcdir_spcube:str, src_exptime:str):
    '''
    :param src_spcube: path to directory where *.tif files are located.
    :param src_exptime: path to exposure times per wavelengthss.

    :returns: `pandas.DataFrame` of paths to .tif files with exposure times.
    '''

    _files = list(pathlib.Path(srcdir_spcube).glob("*.tif"))
    paths_df = pd.DataFrame({'fpath':_files})
    paths_df.loc[:,'wavelength_nm'] = paths_df.loc[:,'fpath'].apply(lambda path: float(re.findall('([0-9]+).tif',str(path))[0]))

    exptime_df = pd.read_csv(src_exptime, delimiter='\t', 
                header=None,
                names=["wavelength_nm", 'exposure_time'])

    exptime_df.loc[:,"wavelength_nm"] = exptime_df.loc[:,"wavelength_nm"].apply(lambda x: float(x))
    exptime_df.loc[:,"_wavelength_ref"] = exptime_df.loc[:,"wavelength_nm"].apply(lambda x: float(x))

    paths_df = pd.merge( paths_df, exptime_df, on='wavelength_nm')
    
    return paths_df

def _load_nuance_spcube(paths_df:pd.DataFrame, norm=True, verbose=False):
    '''
    :param paths_df: must have `fpath`, `wavelength_nm`, and `exposure_time`.
    :returns: spectral cube which shape [H,W, #-spectral-components], and `wavelength` which shapes [#-spectral-components].
    '''
    wl = []
    imgs = []
    for _idx, _row in paths_df.sort_values(by='wavelength_nm').iterrows():
        _wl = _row.loc['wavelength_nm']
        wl.append(_wl)
        fpath = _row.loc['fpath']
        exposure = _row.loc['exposure_time']

        im = PIL.Image.open(fpath)
        imgs.append(np.array(im)/ (exposure if norm else 1.) )
        if verbose: print(f"{_wl} nm, {fpath} -> {im.size}")

    spcube = np.stack(imgs, axis=-1)
    return spcube, np.array(wl)


def load_nuance_spcube(srcdir_spcube:str, src_exptime:str, norm=True, verbose=False):
    '''
    :param src_spcube: path to directory where *.tif files are located.
    :param src_exptime: path to exposure times per wavelengthss.

    :returns: spectral cube which shape [H,W, #-spectral-components], and `wavelength` which shapes [#-spectral-components].
    '''
    paths_df = listup_nuance_spcube_files(srcdir_spcube=srcdir_spcube, src_exptime=src_exptime, )
    spcube, wl = _load_nuance_spcube(paths_df, norm=norm, verbose=verbose)
    
    return spcube, wl,

def selectROI(img, label=None):
    _img_show = img[:,:, :3]
    id_select_roi = f'ROI selection{f" {label}" if label is not None else None}: press key to confirm'
    ROI = cv2.selectROI(id_select_roi, _img_show, fromCenter=False, showCrosshair=False)
    y,x,h,w = ROI
    print(f'ROI: x1={y}, y1={x}, h={h}, w={w}', )

    x1,x2, y1,y2 = int(x), int(x+w), int(y), int(y+h)
    if img.dtype == np.float64 or img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)

    img = cv2.rectangle(img.copy(), (y1, x1), (y2, x2,), (255, 255, 255), 1)
    id_display_selected_roi = "selected ROI"
    cv2.imshow(id_display_selected_roi, img)

    img_crop_show = img[x1:x2, y1:y2] 
    id_display_cropped = "selected ROI, any key to exit"
    cv2.imshow(id_display_cropped, img_crop_show)
    cv2.waitKey(0)
    cv2.destroyWindow(id_select_roi)
    cv2.destroyWindow(id_display_selected_roi)
    cv2.destroyWindow(id_display_cropped)

    return x1,y1, x2,y2

def crop_spectral_reflectance(spref, img_to_show, wl, label, flg_save=False, dstdir='./'):
    x1,y1,x2,y2 = selectROI(img_to_show, label=label)
    spref_roi = spref[x1:x2, y1:y2, :]

    spref_roi = spref_roi.reshape(np.prod(spref_roi.shape[:2]),-1).T
    spectra = spref_roi

    _mean = spectra.mean(axis=1)
    _sd = spectra.std(axis=1)
    plt.plot(wl, _mean, label='mean')
    plt.fill_between(wl, _mean -_sd, _mean +_sd, label='std.', color=[.5,.5,.5,.5])
    plt.xlim( min(wl), max(wl))
    plt.ylim(0,None)
    plt.xlabel("Wavelength, nm")
    plt.ylabel("Spectral reflectance")
    plt.title(f'{label}')
    plt.legend()

    spectra_df = pd.DataFrame(spectra.T, columns=[wl])
    spectra_df['label'] = label
    # spectra_df.loc[:, ('x1','y1','x2','y2')] = x1,y1,x2,y2 # doesnt work
    spectra_df.loc[:,'x1'] = x1
    spectra_df.loc[:,'y1'] = y1
    spectra_df.loc[:,'x2'] = x2
    spectra_df.loc[:,'y2'] = y2

    if flg_save:
        spectra_df.to_csv(pathlib.Path(dstdir).joinpath(f'spectral_reflectance_[{label}].csv'), )
    return spectra_df

def wl_to_ind (spbands, target):
    target_index = np.where(spbands==target, )[0]
    # print(target_index)
    return target_index.astype(int)[0]

def hsi_read(srcpath, w, h, n_spectral_band):
    '''
    :param srcpath: path to binary HSI file to be read.
    :param w,h: width and height of the hsi. `w=400`, `h=320` for SIS, Ebajapan.
    :param n_spectral_band: number of spectral bands of the HSI. `n_pectral_band=81` for SIS, Ebajapan, the bands must be 900--1,700 nm with 10 nm steps.

    :returns: HSI cube array in `np.uint.16` which shapes [H,W, n_bands].

    '''
    with open(srcpath,'rb') as f:
        img = np.fromfile(f,np.uint16,-1)
    img = np.reshape(img, (h, n_spectral_band, w))
    img = np.transpose(img, (0,2,1))

    return img
