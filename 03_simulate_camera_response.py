# %%
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
import os 
from pathlib import Path
import re
import PIL
import cv2
import seaborn as sns 
import copy
import importlib

import spectral_util
importlib.reload(spectral_util)
from spectral_util import *

import fluorescence_util
importlib.reload(fluorescence_util)
from fluorescence_util import *


# %%
# 'board.xlsx'
# 'board_noFlux.xlsx'
# 'flux_20240925_184835.xlsx'
# 'flux-onBoard.xlsx'
# 'lead_noFlux.xlsx'

srcbase = Path("./data/EEM_F-7000_2025-04-11/")
dstdir = Path("./dst/eem/filter")

srcdata = [
    {
        "path": fpath,
        "sample": fpath.stem.split("_")[0],  # 'ABS_20250411' → 'ABS'
        "label": None
    }
    for fpath in srcbase.glob("*.xlsx")
]
srcdata

# %%
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    print(eem)

    plt.figure()
    eem.plot_contour(level=100, show_sample_name=True)

# %%
em_bands = eem.em_bands
ex_bands = eem.ex_bands

# %% [markdown]
# EEMの1次反射・n次散乱光除去のためのスニペット (クラスメソッドに実装済み)

# %%
verbose = True
_df = copy.deepcopy(eem.eem_df)

def find_nearest(array: np.array, value):
    idx = (np.abs(array - value)).argmin()
    # val = array[idx]
    return int(idx)

def is_out_of_range(value, array: np.array = eem.ex_bands, step=eem.ex_band_step, verbose=False):
    if value+(step/2) < array.min():
        if verbose: print(f"{array.max()} < {value-(step/2)}") 
        return True
    elif array.max() < value-(step/2):
        if verbose: print(f"{array.max()} < {value-(step/2)}")
        return True
    
    return False

def _calc_shift_band(wl_src, degree=1, shift:int=0, band_step=eem.em_band_step):
    ''' eemのemissionの波長からexcitationの1次反射とn次散乱光の波長を計算する
    FIXME 後でGPTに埋めさせる
    :param wl_src: 
    :param degree:
    :param shift:
    :param band_step:

    :returns: 
    '''
    shifted_bands = wl_src * degree+(shift *band_step)

    return shifted_bands

def _elliminate_eem(eem_df, em, step, degree=1, bands_ex:np.array=eem.ex_bands, *, inplace=False, verbose=False):
    '''EEMにおける1次反射と2次散乱光を `np.nan` で埋めて消去する。
    FIXME 後でGPTに埋めさせる
    '''
    if not inplace:
        eem_df = eem_df.copy()

    wl_elim_ex = _calc_shift_band(em, shift= step, degree=degree)
    out_of_range = is_out_of_range(wl_elim_ex, verbose=verbose) 
    if verbose: 
        print(f"base={em} nm:/t({em}, {wl_elim_ex}), out-of-range={out_of_range}")
    idx_elim_ex = find_nearest(bands_ex, wl_elim_ex) # 最も近いexの波長を求める
    wl_elim_ex = bands_ex[idx_elim_ex]
    
    # 最近傍だと範囲を大きく超えていても最大値に張り付く可能性があるので 
    # (600nmの2次光の1200nmでも800nmがexの最大値なので800nmが削除対象に含まれている可能性がある) 
    # 範囲外の波長を除外する

    if em in eem.em_bands and wl_elim_ex in eem.ex_bands and not out_of_range:
        eem_df.loc[em, wl_elim_ex] = np.nan

    return eem_df

def remove_self_reflection_and_scattering_from_eem(eem_df, em_bands=eem.em_bands, margin_steps=3, *, inplace=False, verbose=False):
    if not inplace:
        eem_df = eem_df.copy()

    bands_targets = em_bands
    for target_em in bands_targets:

        for step in range(margin_steps):
            # 1次反射
            eem_df = _elliminate_eem(eem_df, em=target_em, step = -step, degree=1, inplace=inplace, verbose=verbose)
            eem_df = _elliminate_eem(eem_df, em=target_em, step = +step, degree=1, inplace=inplace, verbose=verbose)
            # 2次散乱
            eem_df = _elliminate_eem(eem_df, em=target_em, step = -step, degree=2, inplace=inplace, verbose=verbose)
            eem_df = _elliminate_eem(eem_df, em=target_em, step = +step, degree=2, inplace=inplace, verbose=verbose)
            # 3次散乱
            eem_df = _elliminate_eem(eem_df, em=target_em, step = -step, degree=3, inplace=inplace, verbose=verbose)
            eem_df = _elliminate_eem(eem_df, em=target_em, step = +step, degree=3, inplace=inplace, verbose=verbose)

    return eem_df


# %%
remove_self_reflection_and_scattering_from_eem(eem_df=_df, margin_steps=6, inplace=True)
plt.imshow(_df[eem.ex_bands].values)
# _df.values
_df[eem.ex_bands]

# %% [markdown]
# 1次反射・n次散乱の除去 (クラスメソッドを使用)

# %%
eem.remove_self_reflection_and_scattering_from_eem(margin_steps=6, inplace=True, )
eem.plot_heatmap()
plt.title(eem.sample)

# %%
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    print(eem)

    plt.figure()
    # eem.plot_contour(level=100, show_sample_name=True)

    eem.remove_self_reflection_and_scattering_from_eem(margin_steps=6, inplace=True, )
    eem.plot_heatmap()

    plt.title(eem.sample)



# %%
sample_data = []

for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    print(eem)

    # plt.figure()
    # eem.plot_contour(level=100, show_sample_name=True)

    eem.remove_self_reflection_and_scattering_from_eem(margin_steps=6, inplace=True, )
    
    eem_matrix = eem.mat
    # eem_df = eem.eem_df
    
    # サンプルごとにnumpy配列に追加
    sample_data.append(eem_matrix)

    print(eem_matrix)

# numpy配列に保存
eem_array = np.array(sample_data)

# [放射波長，励起波長，9種類]の形で保存
np.save('eem_data.npy', eem_array)

# 形状確認
print(f'EEM data shape: {eem_array.shape}')

# %% [markdown]
# ---

# %% [markdown]
# # LED SPD

# %%
# _param_set = '200to600nm_narrow'
_param_set = '200to600nm_wide'
_params = {
    '200to600nm_narrow':{'wl_start': 200, 'wl_end': 600, 'band_steps': 10, 'band_resolution': 5, 'wl_FWHM': 10},
    '200to600nm_wide':  {'wl_start': 200, 'wl_end': 600, 'band_steps': 10, 'band_resolution': 5, 'wl_FWHM': 50},
    }
param = _params[_param_set]

_wl_bands = np.arange(param['wl_start'], param['wl_end'], param['band_steps'])
_df = pd.DataFrame({'wavelength':_wl_bands, })
_df.loc[:,_wl_bands] = 0

for wl in _wl_bands:
    _df.loc[_df.loc[:,'wavelength']==wl,wl] = 1

wl_sensor = _wl_bands


# %%
wl_FWHM = param['wl_FWHM']
var_FEHM = pow(wl_FWHM/2.35, 2)
f"σ={var_FEHM}, for {wl_FWHM} nm of FWHM"

# %%
wl_start = param['wl_start']
wl_end = param['wl_end']
wl_res = param.get('band_resolution',5)
wl = np.array(range(wl_start, wl_end+1, wl_res))
wl_width = wl_end - wl_start
wl_bandstep = param['band_steps']
# wl_peeks = np.array(
#     range(wl_start+round(wl_bandstep/2), wl_end+1, wl_bandstep))
wl_peeks = np.array(range(wl_start, wl_end+1, wl_bandstep))
print("Range: {0}--{1}, ".format(wl_peeks[0],
      wl_peeks[-1]), "Band steps: {}".format(wl_bandstep))

label_wl = '{0}to{1}nm'.format(wl_start, wl_end, )

print(np.array(wl_peeks))
print('# of LEDs: {}'.format(len(wl_peeks)))


# %% [markdown]
# ## Make SPD

# %%
spds_fill_ = np.array([
    np.exp(-pow(wl - wl_peek_, 2) / var_FEHM)
    for wl_peek_ in wl_peeks
    ])
spds_fill = spds_fill_.T
spds_fill.shape

# %%
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
plt.plot(
    wl, spds_fill, label=[
    'SPDs'if i == 0 else '_nolegend_' for i in range(spds_fill.shape[1])])
plt.xlim([wl[0], wl[-1]])
plt.ylim([0, plt.ylim()[1]])
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Power')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1.15))


# %%
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
plt.plot(
    wl, spds_fill[:,13], label=[
    'SPDs'if i == 0 else '_nolegend_' for i in [7,]]
# , linewidth = 8
    )
# plt.xlim([wl[0], wl[-1]])
plt.xlim([300, 360])
plt.ylim([0, plt.ylim()[1]])
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Power')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1.15))

spds_fill.shape
wl_peeks

# %% [markdown]
# ----

# %% [markdown]
# # camera sensitivity

# %%
import numpy as np
import matplotlib.pyplot as plt

# 波長範囲の設定（動的に変更可能）
wl_start = 200  # nm（変更したい開始波長）
wl_end = 600   # nm（変更したい終了波長）
wl_step = 5     # nm（分解能）

# 波長範囲の生成
wavelengths = np.arange(wl_start, wl_end + 1, wl_step)

# チャネルごとの設定
channels = [
    {"peak": 500, "fwhm": 380, "max_val": 1.0},
    {"peak": 230, "fwhm": 40, "max_val": 0.3}
]

# 感度行列（列ごとにチャネル）
sensitivity_all = []

for ch in channels:
    sigma = ch["fwhm"] / 2.355  # FWHMから標準偏差を計算
    sensitivity = np.exp(-0.5 * ((wavelengths - ch["peak"]) / sigma) ** 2)  # ガウス分布
    sensitivity /= np.max(sensitivity)  # 最大値で正規化
    sensitivity *= ch["max_val"]  # 最大感度値でスケーリング
    sensitivity_all.append(sensitivity)

# スタック（shape: (len(wl), 2)）
sensitivity_all = np.stack(sensitivity_all, axis=1)

# 合成：単純加算 → 正規化
camera_sensitivity = np.sum(sensitivity_all, axis=1)
camera_sensitivity /= np.max(camera_sensitivity)

# 描画
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, sensitivity_all[:, 0], label='Channel 1 (500nm peak)', linestyle='--')
plt.plot(wavelengths, sensitivity_all[:, 1], label='Channel 2 (250nm peak)', linestyle='--')
plt.plot(wavelengths, camera_sensitivity, label='camera Sensitivity', color='black', linewidth=2)
plt.title('camera Spectral Sensitivity')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Sensitivity')
plt.grid(True)
plt.legend()
plt.xlim(wl_start, wl_end)  # 波長範囲を設定
plt.ylim(0, 1.05)
plt.show()

# 合成感度の形状を確認
camera_sensitivity.shape
# wavelengths

# %% [markdown]
# ---

# %% [markdown]
# # カメラ側にハイパスフィルタを適用

# %%
def generate_cutoff_list(peak_wavelengths, fwhm, margin_step, step):
    """
    各LEDに対応するハイパスフィルタのカットオフ波長を計算して返す。

    Parameters:
        peak_wavelengths (list or np.ndarray): 各LEDの中心波長
        fwhm (float): LEDのFWHM（半値幅）
        margin_step (float): 自己反射回避マージンのステップサイズ
        margin_num (int): ステップ数（margin_step × margin_numが追加マージン）

    Returns:
        list of float: 各LEDに対応したカットオフ波長（nm）
    """
    margin = margin_step * step
    cutoff_list = np.array([pw + fwhm + margin for pw in peak_wavelengths])
    
    return cutoff_list

cutoff_list = generate_cutoff_list(wl_peeks, fwhm=20, margin_step=6, step=5)
# cutoff_list の長さは spds_fillの列数と一致させる
if len(cutoff_list) != spds_fill.shape[1]:
    raise ValueError("cutoff_list length must match number of LEDs")

print(f"wl_peeks:{wl_peeks}")
print(f"cutoff_list:{cutoff_list}")



# %%
def generate_hp_filter_matrix(wavelengths, cutoff_list, transition_width):
    hp_matrix = np.zeros((len(wavelengths), len(cutoff_list)))
    for i, cutoff in enumerate(cutoff_list):
        hp_matrix[:, i] = 1 / (1 + np.exp(-(wavelengths - cutoff) / transition_width))
    return hp_matrix

def plot_effective_camera_sens_by_peak(peak_wavelength, wl_peaks, wavelengths,
                                       camera_sensitivity, hp_filter_matrix, spds_fill,
                                       cutoff_list):
    """
    指定したLEDのピーク波長に対応するindexの有効カメラ感度をプロット。
    """
    idx = np.argmin(np.abs(np.array(wl_peaks) - peak_wavelength))
    cutoff = cutoff_list[idx]

    plt.figure(figsize=(10, 5))

    # 有効感度
    plt.plot(wavelengths, effective_camera_sens[:, idx], label=f'Effective Sens (LED @ {wl_peaks[idx]}nm)', linewidth=2)
    
    # ハイパスフィルタ
    plt.plot(wavelengths, hp_filter_matrix[:, idx], label='High-pass Filter', linestyle='dashed')
    
    # カメラ感度
    plt.plot(wavelengths, camera_sensitivity, label='Camera Sensitivity', color='gray', alpha=0.7)
    
    # LED SPD
    if spds_fill is not None:
        plt.plot(wavelengths, spds_fill[:, idx], label='LED SPD', linestyle='dashdot', alpha=0.7)

    # --- カットオフ波長の縦線 ---
    plt.axvline(cutoff, color='red', linestyle='dotted', linewidth=3, label=f'Cutoff = {cutoff:.1f} nm')
    
    plt.title(f'Effective Camera Sensitivity (LED peak {wl_peaks[idx]} nm)')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Sensitivity / Intensity")
    plt.grid(True)
    plt.legend()
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

# 事前定義済みの変数:
# wavelengths, cutoff_list, camera_sensitivity, wl_peaks, spds_fill

# フィルタパラメータ
transition_width = 10  # nm
hp_filter_matrix = generate_hp_filter_matrix(wavelengths, cutoff_list, transition_width)
effective_camera_sens = camera_sensitivity[:, None] * hp_filter_matrix

# === 使用例 ===
plot_effective_camera_sens_by_peak(
    peak_wavelength=330,  # ここを任意で変える
    wl_peaks=wl_peeks,
    wavelengths=wavelengths,
    camera_sensitivity=camera_sensitivity,
    hp_filter_matrix=hp_filter_matrix,
    spds_fill=spds_fill,
    cutoff_list=cutoff_list
)


# %% [markdown]
# ---

# %% [markdown]
# # EEM × LED × カメラ感度

# %%
print(eem_array.shape)  # (10, 81, 81) = (サンプル数（MP），励起，放射)
print(spds_fill.shape)  # (81, 41) = (光強度（各励起波長），LEDの数)

# shapes: (10, 81, 81) @ (81, 41) → (10, 41, 81)
eem_array_clean = np.nan_to_num(eem_array)
fluorescence = np.einsum('sem,el->slm', eem_array_clean, spds_fill)

# 蛍光 × カメラ感度 → カメラが感じる信号
camera_signal = np.einsum('slm,m->sl', fluorescence, camera_sensitivity)

# (samples, leds, emission_wavelengths) × (emission_wavelengths, leds) → (samples, leds)
camera_signal_filtered = np.einsum('slm,ml->sl', fluorescence, camera_sensitivity_matrix)


print(camera_signal.shape)  # (samples, leds)

# %%
import numpy as np
import matplotlib.pyplot as plt

# === サンプル名の取得 ===
sample_names = []
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    sample_names.append(eem.sample)

# === 波長定義 ===
ex_wavelengths = np.linspace(200, 600, spds_fill.shape[0])  # 励起波長
em_wavelengths = np.linspace(200, 600, fluorescence.shape[2])  # 放射波長

# === 各LEDのピーク波長取得 ===
led_peak_wavelengths = ex_wavelengths[np.argmax(spds_fill, axis=0)]

# === 任意指定（表示するサンプル・LED波長）===
sample_name = 'ABS'
sample_idx = sample_names.index(sample_name)
desired_peak_wavelength = 360
led_idx = np.argmin(np.abs(led_peak_wavelengths - desired_peak_wavelength))
print(f"Selected LED {led_idx} with peak wavelength {led_peak_wavelengths[led_idx]:.1f} nm")

# === 該当サンプル・LEDの蛍光スペクトルとLED SPD・カメラ感度取得 ===
fluor = fluorescence[sample_idx, led_idx]  # shape: (81,)
led_spd = spds_fill[:, led_idx]           # shape: (81,)
cam_resp = fluor * camera_sensitivity     # 蛍光 × カメラ感度
cam_sens = camera_sensitivity             # shape: (81,)

# === プロット ===
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左軸：蛍光スペクトル・カメラ応答
ax1.plot(em_wavelengths, fluor, label='Fluorescence', color='blue')
ax1.plot(em_wavelengths, cam_resp, label='Camera Response (Fluorescence × Sensitivity)', color='green', linestyle='--')
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Intensity (Fluorescence / Camera Response)')
ax1.grid(True)
ax1.legend(loc='upper left')

# 右軸：LED SPD・カメラ感度
ax2 = ax1.twinx()
ax2.plot(ex_wavelengths, led_spd, 'r--', alpha=0.6, label='LED SPD')
ax2.plot(em_wavelengths, cam_sens, 'm:', alpha=0.7, label='Camera Sensitivity')
ax2.set_ylabel('LED SPD / Camera Sensitivity')
ax2.legend(loc='upper right')

plt.title(f'Sample: {sample_name}, LED Peak: {led_peak_wavelengths[led_idx]:.1f} nm')
plt.tight_layout()
plt.show()


# %%



