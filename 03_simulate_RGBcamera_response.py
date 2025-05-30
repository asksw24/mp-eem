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

# srcbase = Path("./data/EEM_F-7000_2025-04-11/")
srcbase = Path("./data/EEM_F-7000_2025-05-29/")
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

# %%
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    print(eem)

    plt.figure()

    # ① 散乱ピーク除去
    eem.remove_self_reflection_and_scattering_from_eem(margin_steps=6,
                                                       remove_first_order=True, 
                                                       inplace=True)

    # ② 追加で散乱領域全体を除去
    eem.remove_scatter_regions(inplace=True)

    eem.plot_heatmap()
    plt.title(eem.sample)



# %%
sample_data = []

for data in srcdata:

    eem = fluorescence_util.EEMF7000(data.get('path'))
    print(eem)

    plt.figure()

    # ①散乱ピーク除去
    eem.remove_self_reflection_and_scattering_from_eem(margin_steps=5,
                                                       remove_first_order=False,
                                                        inplace=True)

    # ②追加で散乱領域全体を除去
    eem.remove_scatter_regions_lowerright(inplace=True)

    # nan を 0 に置換
    eem.eem_df = eem.eem_df.fillna(0)

    eem.plot_heatmap()
    plt.title(eem.sample)


    eem_matrix = eem.mat  # numpy配列を取り出す
    eem_matrix = np.nan_to_num(eem_matrix, nan=0.0)  # nanを0に置換
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
, linewidth = 8
    )
# plt.xlim([wl[0], wl[-1]])
plt.xlim([280, 380])
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

# %% [markdown]
# ## RGBカメラ

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
# channels = [
#     {"peak": 500, "fwhm": 380, "max_val": 1.0},
#     {"peak": 230, "fwhm": 40, "max_val": 0.3}
# ]

channels = [
    {"name": "Red",   "peak": 600, "fwhm": 100, "max_val": 0.9},
    {"name": "Green", "peak": 540, "fwhm": 90,  "max_val": 1.0},
    {"name": "Blue",  "peak": 460, "fwhm": 80,  "max_val": 0.8}
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
RGB_sensitivity_all = np.stack(sensitivity_all, axis=1)

# 合成：単純加算 → 正規化
RGB_camera_sensitivity = np.sum(RGB_sensitivity_all, axis=1)
RGB_camera_sensitivity /= np.max(RGB_camera_sensitivity)

# 描画
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, RGB_sensitivity_all[:, 0], label='Channel 1', linestyle='--')
plt.plot(wavelengths, RGB_sensitivity_all[:, 1], label='Channel 2', linestyle='--')
plt.plot(wavelengths, RGB_sensitivity_all[:, 2], label='Channel 3', linestyle='--')
plt.plot(wavelengths, RGB_camera_sensitivity, label='camera Sensitivity', color='black', linewidth=2)
plt.title('camera Spectral Sensitivity')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Sensitivity')
plt.grid(True)
plt.legend()
plt.xlim(wl_start, wl_end)  # 波長範囲を設定
plt.ylim(0, 1.05)
plt.show()

# 合成感度の形状を確認
print(RGB_camera_sensitivity.shape) # (波長（5step），)
print(RGB_sensitivity_all.shape) # (波長（5step），3チャンネル）
# wavelengths


# %% [markdown]
# ---

# %% [markdown]
# # カメラ側にハイパスフィルタを適用

# %% [markdown]
# ## カットオフ波長の計算

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

# cutoff_list = generate_cutoff_list(wl_peeks, fwhm=50, margin_step=6, step=5)
cutoff_list = generate_cutoff_list(wl_peeks, fwhm=45, margin_step=5, step=1)

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

# (81,) * (81, 41) = (81, 41)
effective_camera_sens = RGB_camera_sensitivity[:, None] * hp_filter_matrix

# === 使用例 ===
plot_effective_camera_sens_by_peak(
    peak_wavelength=380,  # ここを任意で変える
    wl_peaks=wl_peeks,
    wavelengths=wavelengths,
    camera_sensitivity=RGB_camera_sensitivity,
    hp_filter_matrix=hp_filter_matrix,
    spds_fill=spds_fill,
    cutoff_list=cutoff_list
)

print(hp_filter_matrix.shape)
print(effective_camera_sens.shape)

# %% [markdown]
# ---

# %% [markdown]
# # EEM × LED × カメラ感度

# %% [markdown]
# ## RGBカメラ

# %%
print(eem_array.shape)  # (10, 81, 81) = (サンプル数（MP），励起，放射)
print(spds_fill.shape)  # (81, 41) = (光強度（各励起波長），LEDの数)

# shapes: (10, 81, 81) @ (81, 41) → (10, 41, 81){(サンプル数（MP），LEDの数，放射（蛍光）)}　
eem_array = np.nan_to_num(eem_array)
fluorescence = np.einsum('sem,el->slm', eem_array, spds_fill)
print(fluorescence.shape) 


print(effective_camera_sens.shape)

# shapes: (10, 41, 81) @ (81, 41) → 
# 蛍光 × カメラ感度(with Filter) → カメラが感じる信号
camera_signals = np.einsum('slm,ml->sl', fluorescence, effective_camera_sens)
print(camera_signals.shape)


# %%
import numpy as np
import matplotlib.pyplot as plt
# plt.use('TkAgg')  # または 'Qt5Agg', 'WebAgg' など

# === サンプル名の取得 ===
sample_names = []
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    sample_names.append(eem.sample)

# === 波長定義 ===
ex_wavelengths = np.linspace(200, 600, spds_fill.shape[0])  # 励起波長
em_wavelengths = np.linspace(200, 600, fluorescence.shape[2])  # 放射波長

# === 各LEDのピーク波長取得 ===
wl_peeks = ex_wavelengths[np.argmax(spds_fill, axis=0)]

# === 任意指定（表示するサンプル・LED波長）===
sample_name = 'PET'
sample_idx = sample_names.index(sample_name)
desired_peak_wavelength = 280
led_idx = np.argmin(np.abs(wl_peeks - desired_peak_wavelength))
print(f"Selected LED {led_idx} with peak wavelength {wl_peeks[led_idx]:.1f} nm")

# === 該当サンプル・LEDの蛍光スペクトルとLED SPD・カメラ感度取得 ===
fluor = fluorescence[sample_idx, led_idx]  # shape: (81,)
led_spd = spds_fill[:, led_idx]           # shape: (81,)
cam_resp = fluor * effective_camera_sens[:, led_idx]     # 蛍光 × カメラ感度
cam_sens = RGB_camera_sensitivity             # shape: (81,)


# === ハイパスフィルタ情報 ===
hp_curve = hp_filter_matrix[:, led_idx]  # shape: (81,)
cutoff = cutoff_list[led_idx]

# === プロット ===
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左軸：蛍光スペクトル・カメラ応答
ax1.plot(em_wavelengths, fluor, label='Fluorescence', color='green', linestyle='--')
ax1.plot(em_wavelengths, cam_resp, label='Camera Response (Fluorescence × Sensitivity)', color='blue')
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Intensity (Fluorescence / Camera Response)')
ax1.grid(True)
ax1.legend(loc='upper left')

# 右軸：LED SPD・カメラ感度・ハイパスフィルタ
ax2 = ax1.twinx()
ax2.plot(ex_wavelengths, led_spd, 'r--', alpha=0.6, label='LED SPD')
ax2.plot(em_wavelengths, cam_sens, 'm:', alpha=0.7, label='Camera Sensitivity')
ax2.plot(em_wavelengths, hp_curve, color='orange', linestyle='-', alpha=0.7, label='High-pass Filter')
ax2.axvline(cutoff, color='orange', linestyle='dotted', linewidth=2, label=f'Cutoff = {cutoff:.1f} nm')
ax2.set_ylabel('LED SPD / Camera Sensitivity / HP Filter')
ax2.legend(loc='upper right')

plt.title(f'Sample: {sample_name}, LED Peak: {wl_peeks[led_idx]:.1f} nm')
plt.tight_layout()
plt.show()


# %% [markdown]
# ---

# %% [markdown]
# # カメラ出力空間plot

# %% [markdown]
# ## 分光感度（複数センサチャネル）　× ハイパスフィルタ × 放射波長

# %%
# print(fluorescence.shape)               # (10, 41, 81){(サンプル数（MP），LEDの数，放射（蛍光）)}
# print(sensitivity_all.shape)            # (81, 3) = (波長，RGBセンサチャネル)
# print(hp_filter_matrix.shape)

# 有効感度 = 分光感度 × フィルタ
effective_sens = RGB_sensitivity_all[:, :, None] * hp_filter_matrix[:, None, :]
print(effective_sens.shape)

# それぞれのセンサチャネル応答 = 放射波長 × 有効感度
# shapes: (10, 41, 81) @ (3, 81, 41) → (10, 41, 3)
camera_rgb = np.einsum('sle,lcr->slc', fluorescence, effective_sens.transpose(2,1,0))
print(camera_rgb.shape)


# plt.plot(em_wavelengths, camera_rgb[:, 1])

# %%
import matplotlib.pyplot as plt

def plot_sensor_led_filter(wavelengths, sensitivity_all, hp_filter_matrix,
                           effective_sens, spds_fill, led_idx):
    """
    指定したLEDに対して、感度・フィルタ・LEDスペクトルをまとめて表示
    """
    plt.figure(figsize=(10, 6))
    
    # --- 各種成分の取得 ---
    led_wavelength = wl_peeks[led_idx]
    
    # 元のRGB感度（3チャンネル）
    for i, color in enumerate(['r', 'g', 'b']):
        plt.plot(wavelengths, sensitivity_all[:, i],
                 linestyle='dashed', color=color, alpha=0.5, label=f'Original {color.upper()} Sens.')
    
    # ハイパスフィルタ
    plt.plot(wavelengths, hp_filter_matrix[:, led_idx],
             linestyle='dotted', color='black', label='High-pass Filter')

    # 有効感度（フィルタ適用後）
    for i, color in enumerate(['r', 'g', 'b']):
        plt.plot(wavelengths, effective_sens[:, i, led_idx],
                 linestyle='solid', color=color, label=f'Effective {color.upper()} Sens.')

    # LEDスペクトル
    plt.plot(wavelengths, spds_fill[:, led_idx],
             linestyle='dashdot', color='magenta', label=f'LED SPD ({led_wavelength:.0f} nm)')
    
    # --- 描画設定 ---
    plt.title(f"LED @ {led_wavelength:.0f} nm - Sensor Response & Filter")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Sensitivity / Intensity (normalized)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === 使用例 ===
plot_sensor_led_filter(
    wavelengths=wavelengths,
    sensitivity_all=RGB_sensitivity_all,
    hp_filter_matrix=hp_filter_matrix,
    effective_sens=effective_sens,
    spds_fill=spds_fill,
    led_idx=19  # ← 任意のLEDインデックスに変更可能
)


# %% [markdown]
# ## カメラ出力空間・分離度

# %%
# import numpy as np
# import plotly.graph_objs as go
# import gradio as gr
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import pdist, squareform
# import io
# from PIL import Image

# # --- 事前定義前提 ---
# # camera_rgb: shape = (num_samples, num_wavelengths, 3)
# # wl_peeks: shape = (num_wavelengths,)
# # sample_names: list[str] or np.array

# # === サンプル名の取得 ===
# sample_names = []
# for data in srcdata:
#     eem = fluorescence_util.EEMF7000(data.get('path'))
#     sample_names.append(eem.sample)

# # RGB軸のスケーリング
# all_R = camera_rgb[:, :, 0]
# all_G = camera_rgb[:, :, 1]
# all_B = camera_rgb[:, :, 2]

# global_min = min(all_R.min(), all_G.min(), all_B.min())
# global_max = max(all_R.max(), all_G.max(), all_B.max())
# margin = (global_max - global_min) * 0.1
# axis_range = [global_min - margin, global_max + margin]

# def plot_rgb_by_excitation(selected_peak):
#     led_idx = int(np.argmin(np.abs(wl_peeks - selected_peak)))
#     fig = go.Figure()

#     for s, name in enumerate(sample_names):
#         R, G, B = camera_rgb[s, led_idx]
#         fig.add_trace(go.Scatter3d(
#             x=[R], y=[G], z=[B],
#             mode='markers+text',
#             marker=dict(size=6),
#             text=[name],
#             textposition="top center",
#             name=name
#         ))

#     fig.update_layout(
#         width=800,
#         height=700,
#         margin=dict(l=20, r=20, t=50, b=20),
#         scene=dict(
#             xaxis=dict(title='Red', range=axis_range),
#             yaxis=dict(title='Green', range=axis_range),
#             zaxis=dict(title='Blue', range=axis_range),
#             aspectmode="manual",
#             aspectratio=dict(x=1, y=1, z=1),
#             camera=dict(eye=dict(x=2.0, y=-2.0, z=1.0))
#         ),
#         title=f"RGB Responses at Excitation Peak {wl_peeks[led_idx]:.0f} nm"
#     )

#     return fig

# def plot_rgb_distance_heatmap(selected_peak):
#     led_idx = int(np.argmin(np.abs(wl_peeks - selected_peak)))
#     rgb_vectors = camera_rgb[:, led_idx, :]  # shape = (num_samples, 3)
#     dist_matrix = squareform(pdist(rgb_vectors, metric="euclidean"))

#     # 固定スケールのため、全波長分の距離値を使って min/max を取得（必要に応じて）
#     all_dists = []
#     for i in range(camera_rgb.shape[1]):
#         d = squareform(pdist(camera_rgb[:, i, :]))
#         all_dists.append(d)
#     vmin = np.min(all_dists)
#     vmax = np.max(all_dists)

#     # 下三角をマスク（視認性向上）
#     mask = np.tril(np.ones_like(dist_matrix, dtype=bool))

#     fig, ax = plt.subplots(figsize=(6, 5))
#     sns.heatmap(
#         dist_matrix,
#         mask=mask,
#         cmap='viridis',
#         xticklabels=sample_names,
#         yticklabels=sample_names,
#         vmin=vmin,
#         vmax=vmax,
#         square=True,
#         cbar=True,
#         annot=False,
#         ax=ax
#     )
#     ax.set_title(f"RGB Distance Matrix (Excitation {wl_peeks[led_idx]:.0f} nm)")
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf)

# # Gradio UI: Sliderで両方出力
# peak_min = int(wl_peeks.min())
# peak_max = int(wl_peeks.max())
# step = int(np.min(np.diff(wl_peeks)))

# with gr.Blocks(title="RGBセンサ応答と距離行列の同時可視化") as demo:
#     gr.Markdown("### 励起波長を選択して、RGBセンサ応答と距離行列を同時に表示")

#     slider = gr.Slider(peak_min, peak_max, step=step, label="励起波長 [nm]", interactive=True)

#     with gr.Row():
#         rgb_output = gr.Plot(label="3D RGBプロット")
#         heatmap_output = gr.Image(type="pil", label="RGB距離行列ヒートマップ")

#     slider.change(fn=plot_rgb_by_excitation, inputs=slider, outputs=rgb_output)
#     slider.change(fn=plot_rgb_distance_heatmap, inputs=slider, outputs=heatmap_output)

# demo.launch()


# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import io
from PIL import Image
import gradio as gr

# --- サンプル名の取得 ---
sample_names = []
for data in srcdata:
    eem = fluorescence_util.EEMF7000(data.get('path'))
    sample_names.append(eem.sample)

# --- 波長定義とLEDピーク ---
ex_wavelengths = np.linspace(200, 600, spds_fill.shape[0])
em_wavelengths = np.linspace(200, 600, fluorescence.shape[2])
led_peak_wavelengths = ex_wavelengths[np.argmax(spds_fill, axis=0)]


# --- 蛍光スペクトル可視化（matplotlib→PIL Image） ---
def plot_spectra(sample_name, desired_peak_wavelength):
    sample_idx = sample_names.index(sample_name)
    led_idx = np.argmin(np.abs(wl_peeks - desired_peak_wavelength))

    fluor = fluorescence[sample_idx, led_idx]
    led_spd = spds_fill[:, led_idx]
    cam_sens = RGB_camera_sensitivity
    cam_resp = fluor * effective_camera_sens[:, led_idx]
    hp_curve = hp_filter_matrix[:, led_idx]
    cutoff = cutoff_list[led_idx]

    em_wavelengths = np.linspace(200, 600, fluorescence.shape[2])
    ex_wavelengths = np.linspace(200, 600, spds_fill.shape[0])

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(em_wavelengths, fluor, label='Fluorescence', color='green', linestyle='--')
    ax1.plot(em_wavelengths, cam_resp, label='Camera Response (Fluor × Sens)', color='blue')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Intensity')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(ex_wavelengths, led_spd, 'r--', alpha=0.6, label='LED SPD')
    ax2.plot(em_wavelengths, cam_sens, 'm:', alpha=0.7, label='Camera Sensitivity')
    ax2.plot(em_wavelengths, hp_curve, color='orange', linestyle='-', alpha=0.7, label='High-pass Filter')
    ax2.axvline(cutoff, color='orange', linestyle='dotted', linewidth=2, label=f'Cutoff = {cutoff:.1f} nm')
    ax2.set_ylabel('SPD / Sensitivity / Filter')
    ax2.legend(loc='upper right')

    plt.title(f'Sample: {sample_name}, LED Peak: {wl_peeks[led_idx]:.1f} nm')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- RGB 3Dプロット ---
def plot_rgb_by_excitation(selected_peak):
    led_idx = int(np.argmin(np.abs(wl_peeks - selected_peak)))
    fig = go.Figure()

    for s, name in enumerate(sample_names):
        R, G, B = camera_rgb[s, led_idx]
        fig.add_trace(go.Scatter3d(
            x=[R], y=[G], z=[B],
            mode='markers+text',
            marker=dict(size=6),
            text=[name],
            textposition="top center",
            name=name
        ))

    all_R = camera_rgb[:, :, 0]
    all_G = camera_rgb[:, :, 1]
    all_B = camera_rgb[:, :, 2]

    global_min = min(all_R.min(), all_G.min(), all_B.min())
    global_max = max(all_R.max(), all_G.max(), all_B.max())
    margin = (global_max - global_min) * 0.1
    axis_range = [global_min - margin, global_max + margin]

    fig.update_layout(
        width=700,
        height=750,
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis=dict(title='Red', range=axis_range),
            yaxis=dict(title='Green', range=axis_range),
            zaxis=dict(title='Blue', range=axis_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=2.0, y=-2.0, z=1.0))
        ),
        title=f"RGB Responses at Excitation Peak {wl_peeks[led_idx]:.0f} nm"
    )

    return fig

def plot_rgb_separability(selected_peak):
    separability_means = []
    separability_stds = []
    for i in range(camera_rgb.shape[1]):
        responses = camera_rgb[:, i, :]
        distances = pdist(responses, metric='euclidean')
        separability_means.append(np.mean(distances))
        separability_stds.append(np.std(distances))
    separability_means = np.array(separability_means)
    separability_stds = np.array(separability_stds)

    max_idx = np.argmax(separability_means)
    best_wavelength = wl_peeks[max_idx]
    best_score = separability_means[max_idx]

    selected_idx = int(np.argmin(np.abs(wl_peeks - selected_peak)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        wl_peeks,
        separability_means,
        yerr=separability_stds,
        fmt='-o',
        capsize=5,
        label='Separability (mean ± std)'
    )
    ax.axvline(best_wavelength, color='r', linestyle='--', label=f'Best: {best_wavelength:.0f} nm')
    ax.scatter(best_wavelength, best_score, color='red')
    ax.axvline(wl_peeks[selected_idx], color='blue', linestyle='-', linewidth=2, label=f'Selected: {wl_peeks[selected_idx]:.0f} nm')

    ax.set_title('RGB Separability by Excitation Wavelength')
    ax.set_xlabel('Excitation Wavelength [nm]')
    ax.set_ylabel('Separability')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# --- RGB距離ヒートマップ ---
def plot_rgb_distance_heatmap(selected_peak):
    led_idx = int(np.argmin(np.abs(wl_peeks - selected_peak)))
    rgb_vectors = camera_rgb[:, led_idx, :]
    dist_matrix = squareform(pdist(rgb_vectors, metric="euclidean"))

    all_dists = []
    for i in range(camera_rgb.shape[1]):
        d = squareform(pdist(camera_rgb[:, i, :]))
        all_dists.append(d)
    vmin = np.min(all_dists)
    vmax = np.max(all_dists)

    mask = np.tril(np.ones_like(dist_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(dist_matrix, mask=mask, cmap='viridis',
                xticklabels=sample_names, yticklabels=sample_names,
                vmin=vmin, vmax=vmax, square=True, cbar=True, ax=ax)
    ax.set_title(f"RGB Distance Matrix (Excitation {wl_peeks[led_idx]:.0f} nm)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- 統合UI構築 ---
def update_all(sample_name, peak_nm):
    spectrum_img = plot_spectra(sample_name, peak_nm)
    rgb_fig = plot_rgb_by_excitation(peak_nm)
    dist_img = plot_rgb_distance_heatmap(peak_nm)
    separability_img = plot_rgb_separability(peak_nm)  # 追加
    return spectrum_img, rgb_fig, dist_img, separability_img


def build_ui():
    peak_min = int(led_peak_wavelengths.min())
    peak_max = int(led_peak_wavelengths.max())
    step = int(np.min(np.diff(led_peak_wavelengths)))

    with gr.Blocks() as demo:
        gr.Markdown("## 蛍光スペクトル・RGB応答・距離行列・分離度の可視化")

        # 上段
        with gr.Row():
            with gr.Column(scale=1):
                sample_dropdown = gr.Dropdown(choices=sample_names, value=sample_names[0], label="サンプル選択")
                peak_slider = gr.Slider(minimum=wl_peeks.min(), maximum=wl_peeks.max(),
                                        step=1, value=wl_peeks[0], label="励起波長 (LEDピーク) [nm]")
                spectrum_img = gr.Image(label="蛍光スペクトル")

            with gr.Column(scale=1):
                rgb_3d_plot = gr.Plot(label="RGB 3Dプロット")

        # 下段：距離行列（左）＋ 分離度（右）
        with gr.Row():
            with gr.Column(scale=1):
                distance_heatmap = gr.Image(label="RGB距離行列ヒートマップ")

            with gr.Column(scale=1):
                separability_img = gr.Image(label="分離度プロット")

        # イベント
        sample_dropdown.change(fn=update_all, inputs=[sample_dropdown, peak_slider],
                            outputs=[spectrum_img, rgb_3d_plot, distance_heatmap, separability_img])
        peak_slider.change(fn=update_all, inputs=[sample_dropdown, peak_slider],
                        outputs=[spectrum_img, rgb_3d_plot, distance_heatmap, separability_img])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()



