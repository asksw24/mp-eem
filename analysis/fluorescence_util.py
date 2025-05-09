import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pathlib
import re

import spectral_util
import importlib
importlib.reload(spectral_util)
# from spectral_util import *

def load_f7000_spectral_measurement(srcpath:pathlib.Path, sample:str, label:str=None, *, verbose=False):
    df = pd.read_excel(srcpath, header=None, index_col=0)
    df = df.T

    df['sample'] = sample
    df['label'] = label if label is not None else df['sample']
    EX_WAVE_BAND = '励起波長:'
    EM_START_WAVE_BAND = '蛍光開始波長:'
    EM_END_WAVE_BAND = '蛍光終了波長:'
    # EX_BAND_STEP = '励起側ｽﾘｯﾄ:'
    # EM_BAND_STEP = '蛍光側ｽﾘｯﾄ:'

    df.loc[:, EM_START_WAVE_BAND] = df.loc[:, EM_START_WAVE_BAND].apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
    df.loc[:, EM_END_WAVE_BAND] = df.loc[:, EM_END_WAVE_BAND].apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
    
    bands = _get_bands(df=df, )
    if verbose:
        print(f"bands: {bands}")

    return df, bands

def _get_bands(
        df, 
        *,
        EM_START_WAVE_BAND = '蛍光開始波長:',
        EM_END_WAVE_BAND = '蛍光終了波長:',
    ):
    f'''
    :param: df: 1 column `{pd.DataFrame}` or `{pd.Series}`.
    '''
    if isinstance(df, pd.DataFrame):
        _vars = df.columns.tolist()
        bands = [k for k in _vars if type(k)==int and df[ EM_START_WAVE_BAND].dropna().values <= k and k <= df.loc[:,EM_END_WAVE_BAND].dropna().values]
    elif isinstance(df, pd.Series):
        _vars = df.index.tolist()
        bands = [k for k in _vars if type(k)==int and df[EM_START_WAVE_BAND]<= k and k <= df.loc[EM_END_WAVE_BAND]]
    else:
        raise ValueError(f'`df` must be {pd.DataFrame} or {pd.Series}')
    
    return bands

class EEMF7000():
    EX_START_WAVE_BAND_COL  = '励起開始波長:'
    EX_END_WAVE_BAND_COL    = '励起終了波長:'
    EX_BAND_STEP_COL        = '励起側ｻﾝﾌﾟﾘﾝｸﾞ間隔:'

    EM_START_WAVE_BAND_COL  = '蛍光開始波長:'
    EM_END_WAVE_BAND_COL    = '蛍光終了波長:'
    EM_BAND_STEP_COL        = '蛍光側ｻﾝﾌﾟﾘﾝｸﾞ間隔:'
    
    SAMPLE_COL_NAME     = "ｻﾝﾌﾟﾙ:"
    EEM_DATA_START_COL  = 'ﾃﾞｰﾀﾘｽﾄ'
    EX_BANDS_COL = 'ex_bands'
    EM_BANDS_COL = 'em_bands'

    def __init__(
            self, 
            srcpath:pathlib.Path, 
            *, 
            ex_min_plot=None, ex_max_plot=None,
            em_min_plot=None, em_max_plot=None,
            sample:str=None, label:str=None, verbose=False,
            **kwds,
            ):

        df = self.load_f7000_eem_measurement(srcpath=srcpath, sample=sample, label=label, verbose=verbose)
        eem = self.set_eem(df)

        self.ex_min_plot = ex_min_plot
        self.ex_max_plot = ex_max_plot
        self.em_min_plot = em_min_plot
        self.em_max_plot = em_max_plot


    def __str__(self):
        return f'EEM of "{self.sample}"-sample in Ex=[{self.ex_bands.min()}--{self.ex_bands.max()}nm of {self.ex_band_step}nm steps] and Em=[{self.em_bands.min()}--{self.em_bands.max()}nm of {self.em_band_step}nm steps].'
    
    def __repr__(self):
        return f'EEM of "{self.sample}"-sample'
        

    def load_f7000_eem_measurement(self, srcpath:pathlib.Path, *, sample:str=None, label:str=None, verbose=False):
        df = pd.read_excel(srcpath, header=None, index_col=0)
        df = df.T

        if sample is None:
            sample = df.iloc[0].loc[self.SAMPLE_COL_NAME]
        self.sample = sample

        df['sample']    = sample
        df['label']     = label if label is not None else df['sample']
        
        self.ex_start_band  = df.loc[:, self.EX_START_WAVE_BAND_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        self.ex_end_band    = df.loc[:, self.EX_END_WAVE_BAND_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        self.ex_band_step    = df.loc[:, self.EX_BAND_STEP_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        df.loc[:, self.EX_START_WAVE_BAND_COL]   = self.ex_start_band = self.ex_start_band.values[0]
        df.loc[:, self.EX_END_WAVE_BAND_COL]     = self.ex_end_band   = self.ex_end_band.values[0]
        df.loc[:, self.EX_BAND_STEP_COL]         = self.ex_band_step  = self.ex_band_step.values[0]

        self.em_start_band = df.loc[:, self.EM_START_WAVE_BAND_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        self.em_end_band = df.loc[:, self.EM_END_WAVE_BAND_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        self.em_band_step = df.loc[:, self.EM_BAND_STEP_COL].dropna().apply(lambda x: float(re.findall(r"([0-9.]+) nm", x)[0]))
        df.loc[:, self.EM_START_WAVE_BAND_COL]   = self.em_start_band = self.em_start_band.values[0]
        df.loc[:, self.EM_END_WAVE_BAND_COL]     = self.em_end_band   = self.em_end_band.values[0]
        df.loc[:, self.EM_BAND_STEP_COL]         = self.em_band_step  = self.em_band_step.values[0]
        
        self.ex_bands = np.concatenate([
            np.arange(self.ex_start_band, self.ex_end_band, self.ex_band_step),
            [self.ex_end_band]]
            )
        self.em_bands = np.concatenate([
            np.arange(self.em_start_band, self.em_end_band, self.em_band_step),
            [self.em_end_band]]
            )
        columns = df.columns.tolist()
        columns[columns.index(self.EEM_DATA_START_COL)+1] = self.EM_BANDS_COL
        df.columns  = columns
        em_bands    = df.loc[:, self.EM_BANDS_COL]

        if verbose:
            print(f"Excitation: {self.ex_bands}")
            print(f"Emission: {self.em_bands}")

        self.df = df
        return df

    def set_eem(self, df=None):
        if df is None:
            df = self.df

        em_bands = df.loc[:, self.EM_BANDS_COL]
        
        eem_df = df.loc[:, self.ex_bands].copy()
        eem_df.loc[:, self.EM_BANDS_COL] = em_bands
        eem_df.set_index(em_bands.astype(float), inplace=True, append=False)
        eem_df.rename({wl: float(wl) for wl in eem_df.columns if type(wl) != str} , axis='columns', inplace=True)
        for col in eem_df.columns:
            eem_df[col] = pd.to_numeric(eem_df.loc[:, col], errors='coerce')

        self.eem_df = eem_df

        return eem_df

    def plot_contour(self, vmax=None, level=None, title=None, show_sample_name=False):
        plt.figure()
        plt.contour(self.ex_bands, self.em_bands, self.mat, level, vmax=vmax, vmin=0)
        plt.axis('square')
        plt.ylabel('Excitation [nm]')
        plt.xlabel('Emission [nm]')
        plt.colorbar()

        em_bands = self.em_bands
        ex_bands = self.ex_bands

        if show_sample_name:
            plt.title(self.sample)

    def plot_pcolor(self, *,
            vmax=None,
            ex_min=None, ex_max=None, 
            em_min=None, em_max=None, 
            title=None, show_sample_name=False,
            ):
        ex_bands, em_bands = self.bands_limited(
            ex_min=ex_min, ex_max=ex_max,
            em_min=em_min, em_max=em_max,
        )

        df = self.mat_df.loc[ex_bands, em_bands]
        
        plt.pcolor(df.values, cmap='viridis', vmax=vmax)
        ytk = np.arange(0, ex_bands.shape[0], 5)
        xtk = np.arange(0, em_bands.shape[0], 5)
        plt.yticks(ytk+0.5, ex_bands[ytk].astype(int), minor=False)
        plt.xticks(xtk+0.5, em_bands[xtk].astype(int), minor=False, rotation='vertical')
        plt.axis('square', )
        plt.ylim(0, ex_bands.shape[0]-1)
        plt.xlim(0, em_bands.shape[0]-1)
        plt.ylabel('ex [nm]')
        plt.xlabel('em [nm]')

        if show_sample_name:
            plt.title(self.sample)

    def plot_heatmap(self,
            vmax=None, title=None, *,
            ex_min=None, ex_max=None, 
            em_min=None, em_max=None, 
            cmap="viridis",
            show_sample_name=False, omit_under_decimal_point_in_ticks=True,
            ax=None,
            ):

        if ax is None:
            ax = plt.gca()
        eem_mat_df = self.mat_df
        
        ex_bands = sorted(self.ex_bands)[::-1]
        em_bands = sorted(self.em_bands)

        ex_bands, em_bands = self.bands_limited(
            ex_bands=ex_bands,em_bands=em_bands, 
            ex_min=ex_min, ex_max=ex_max,
            em_min=em_min, em_max=em_max,
            )

        ax = sns.heatmap(eem_mat_df.loc[ex_bands,em_bands], square=True, vmax=vmax, cmap=cmap,ax=ax)
        ax.set_ylabel('Excitation [nm]')
        ax.set_xlabel('Emission [nm]')
        
        if omit_under_decimal_point_in_ticks:
            xtick_labels = [round(float(label.get_text())) if len(label.get_text())>0 else None for label in ax.get_xticklabels() ]
            ytick_labels = [round(float(label.get_text())) if len(label.get_text())>0 else None for label in ax.get_yticklabels() ]
            ax.set_xticklabels(xtick_labels) if len(xtick_labels)>0 else None
            ax.set_yticklabels(ytick_labels) if len(ytick_labels)>0 else None

        if show_sample_name:
            ax.set_title(self.sample)

    def bands_limited(self, *,
            ex_bands:np.ndarray=None, em_bands:np.ndarray=None,
            ex_min:np.ndarray=None, ex_max:np.ndarray=None, 
            em_min:np.ndarray=None, em_max:np.ndarray=None, 
            ):
        ex_bands = ex_bands if ex_bands is not None else self.ex_bands
        em_bands = em_bands if em_bands is not None else self.em_bands

        ex_bands = np.array(ex_bands)
        em_bands = np.array(em_bands)

        ex_min = ex_min if ex_min is not None else self.ex_min_plot
        ex_max = ex_max if ex_max is not None else self.ex_max_plot
        em_min = em_min if em_min is not None else self.em_min_plot
        em_max = em_max if em_max is not None else self.em_max_plot
        
        ex_min = ex_bands.min() if ex_min is None else ex_min
        ex_max = ex_bands.max() if ex_max is None else ex_max
        em_min = em_bands.min() if em_min is None else em_min
        em_max = em_bands.max() if em_max is None else em_max

        ex_bands = ex_bands[ np.logical_and(ex_min <= ex_bands, ex_bands <= ex_max) ]
        em_bands = em_bands[ np.logical_and(em_min <= em_bands, em_bands <= em_max) ]
        
        return ex_bands, em_bands

    @property
    def mat_df(self):
        return self.eem_df.loc[self.em_bands, self.ex_bands]
    
    @property 
    def mat(self):
        return self.eem_df.loc[:, self.ex_bands].values
    
    @property
    def melt_df(self):
        return self.eem_df.melt(value_vars=self.ex_bands, value_name='intensity', id_vars=[self.EM_BANDS_COL], var_name=self.EX_BANDS_COL)

    ### EEMの自己反射と散乱光を除去する関係のメソッド群
    def find_nearest(array: np.array, value):
        idx = (np.abs(array - value)).argmin()
        # val = array[idx]
        return int(idx)

    def is_out_of_range(self, value, array: np.array = None, step=None, verbose=False):
        array = self.ex_bands if array is None else array
        step = self.ex_band_step if step is None else step

        if value+(step/2) < array.min():
            if verbose: print(f"{value+(step/2)} < {array.min()}") 
            return True
        elif array.max() < value-(step/2):
            if verbose: print(f"{array.max()} < {value-(step/2)}")
            return True
        
        return False

    def _calc_shift_band(self, wl_src, degree=1, shift:int=0, band_step:np.array=None):
        ''' eemのemissionの波長からexcitationの1次反射とn次散乱光の波長を計算する
        FIXME 後でGPTに埋めさせる
        :param wl_src: 
        :param degree:
        :param shift:
        :param band_step:

        :returns: 
        '''
        band_step = self.em_band_step if band_step is None else band_step
        shifted_bands = wl_src * degree+(shift *band_step)

        return shifted_bands

    def _elliminate_eem(self, eem_df, em, step, degree=1, bands_ex:np.array=None,
                        *, 
                        inplace=False, verbose=False
                        ):
        '''EEMにおける1次反射と2次散乱光を `np.nan` で埋めて消去する。
        FIXME 後でGPTに埋めさせる
        '''
        bands_ex = self.ex_bands if bands_ex is None else bands_ex

        if not inplace:
            eem_df = eem_df.copy()

        wl_elim_ex = self._calc_shift_band(em, shift= step, degree=degree)
        out_of_range = self.is_out_of_range(wl_elim_ex, verbose=verbose) 
        if verbose: 
            print(f"base={em} nm:\t({em}, {wl_elim_ex}), out-of-range={out_of_range}")
        idx_elim_ex = EEMF7000.find_nearest(bands_ex, wl_elim_ex) # 最も近いexの波長を求める
        wl_elim_ex = bands_ex[idx_elim_ex]
        
        # 最近傍だと範囲を大きく超えていても最大値に張り付く可能性があるので 
        # (600nmの2次光の1200nmでも800nmがexの最大値なので800nmが削除対象に含まれている可能性がある) 
        # 範囲外の波長を除外する
        if em in self.em_bands and wl_elim_ex in self.ex_bands and not out_of_range:
            eem_df.loc[em, wl_elim_ex] = np.nan

        return eem_df

    def remove_self_reflection_and_scattering_from_eem(
            self, em_bands:np.array=None, margin_steps=3, 
            *, 
            inplace=False, verbose=False
            ):
        eem_df = self.eem_df
        em_bands = self.em_bands if em_bands is None else em_bands
        if not inplace:
            eem_df = eem_df.copy()

        bands_targets = em_bands
        for target_em in bands_targets:

            for step in range(margin_steps):
                # 1次反射
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = -step, degree=1, inplace=inplace, verbose=verbose)
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = +step, degree=1, inplace=inplace, verbose=verbose)
                # 2次散乱
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = -step, degree=2, inplace=inplace, verbose=verbose)
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = +step, degree=2, inplace=inplace, verbose=verbose)
                # 3次散乱
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = -step, degree=3, inplace=inplace, verbose=verbose)
                eem_df = self._elliminate_eem(eem_df, em=target_em, step = +step, degree=3, inplace=inplace, verbose=verbose)

        return eem_df
    ### 