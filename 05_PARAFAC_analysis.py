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
print(Path.cwd())
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
# # 前処理

# %% [markdown]
# ## 散乱光の除去と波長域の調整

# %%
# %% EEMデータの読み込みと前処理（散乱除去 & NaN保持）

sample_data_processed = []
sample_name = []

# 波長域を先に定義しておく
eem_template = fluorescence_util.EEMF7000(srcdata[0]['path'])
ex_bands_full = eem_template.ex_bands
em_bands_full = eem_template.em_bands

# 250nm以上の波長マスクを作成
ex_mask = ex_bands_full >= 250
em_mask = em_bands_full >= 250

# トリミング後の波長域を保存
ex_bands_trimmed = ex_bands_full[ex_mask]
em_bands_trimmed = em_bands_full[em_mask]


print("--- データの前処理を開始します ---")
for data in srcdata:
    # EEMデータの読み込み
    eem = fluorescence_util.EEMF7000(data.get('path'))

    # ① 散乱ピーク除去（NaNを代入）
    eem.remove_self_reflection_and_scattering_from_eem(margin_steps=6,
                                                       remove_first_order=True,
                                                       inplace=True)
    # ② 追加で散乱領域全体を除去（NaNを代入）
    eem.remove_scatter_regions(inplace=True)

    # 生の行列を取得
    eem_matrix = eem.mat

    # ③ 波長域をトリミング
    eem_matrix_trimmed = eem_matrix[np.ix_(ex_mask, em_mask)]

    # 前処理済みのデータをリストに追加
    sample_data_processed.append(eem_matrix_trimmed)
    sample_name.append(eem.sample)
    print(f"  - サンプル '{eem.sample}' の前処理が完了しました。")

print("--- 全ての前処理が完了しました ---")

# リストを3Dのnumpy配列に変換
# この段階ではNaNが含まれている
eem_array_processed = np.array(sample_data_processed)

print("\n処理後のデータ形状:", eem_array_processed.shape)
print("励起波長の数:", len(ex_bands_trimmed))
print("蛍光波長の数:", len(em_bands_trimmed))

# %%
sample_name

# %% [markdown]
# ## 正規化

# %%
# %% データの正規化（論文準拠: Unit Norm Scaling）

# 正規化済みのデータを格納する新しいリストを作成
sample_data_normalized = []

print("--- データの正規化を開始します（論文準拠）---")
for i, eem_matrix in enumerate(eem_array_processed):
    # NaNを無視して、二乗和を計算
    sum_of_squares = np.nansum(eem_matrix**2)

    # 二乗和が0または非常に小さい場合は、ゼロ除算を避ける
    if sum_of_squares > 1e-8:
        eem_normalized = eem_matrix / sum_of_squares
    else:
        # データが全て0やNaNの場合、そのまま（変更なし）
        eem_normalized = eem_matrix

    sample_data_normalized.append(eem_normalized)
    print(f"  - サンプル '{sample_name[i]}' の正規化が完了しました。")

print("--- 全ての正規化が完了しました ---")

# 正規化後のデータを3Dのnumpy配列に変換
# この段階でもNaNは保持されています
eem_array_normalized = np.array(sample_data_normalized)

print("\n正規化後のデータ形状:", eem_array_normalized.shape)

# %%
# %% ステップA: サンプル名とデータの対応を確認

print("--- サンプル名とデータの対応を確認します ---")
print(f"合計サンプル数: {len(sample_name)}")
print("-" * 40)

for i, name in enumerate(sample_name):
    # 対応するデータの形状も一緒に表示
    data_shape = eem_array_normalized[i].shape
    print(f"インデックス {i}: サンプル名 = '{name}', データ形状 = {data_shape}")

print("-" * 40)
print("上記の順番でデータは並んでいます。")

# %%
# %% ステップA: サンプル名とデータの対応を確認

print("--- サンプル名とデータの対応を確認します ---")
print(f"合計サンプル数: {len(sample_name)}")
print("-" * 40)

for i, name in enumerate(sample_name):
    # 対応するデータの形状も一緒に表示
    data_shape = eem_array_normalized[i].shape
    print(f"インデックス {i}: サンプル名 = '{name}', データ形状 = {data_shape}")

print("-" * 40)
print("上記の順番でデータは並んでいます。")

# %% [markdown]
# ---

# %% [markdown]
# # 妥当性評価

# %%
max_components = 2

# %%
# %% ステップB: 横軸を見やすくしたレバレッジプロット（レイアウト調整版）

import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import numpy as np
import matplotlib.pyplot as plt

# --- この部分は変更ありません ---
eem_array_imputed = np.nan_to_num(eem_array_normalized, nan=0.0)
tensor = tl.tensor(np.transpose(eem_array_imputed, (1, 2, 0)))
# max_components = 9
component_range = range(2, max_components + 1)
leverage_results = {}
for n_comp in component_range:
    weights, factors = non_negative_parafac(tensor,
                                            rank=n_comp, n_iter_max=200,
                                            tol=1e-6, init='random')
    sample_scores = factors[2]
    try:
        q, _ = np.linalg.qr(sample_scores)
        leverage = np.sum(q**2, axis=1)
        leverage_results[n_comp] = leverage
    except np.linalg.LinAlgError:
        leverage_results[n_comp] = np.full(tensor.shape[2], np.nan)
# --- ここまで変更なし ---


# --- 結果の可視化（レイアウト調整版） ---
print("\n--- レバレッジをプロットします ---")

# ▼▼▼ グラフ全体のサイズを、より縦長に調整 ▼▼▼
fig, axes = plt.subplots(len(component_range), 1, figsize=(12, 5 * len(component_range)), sharex=True)
if len(component_range) == 1:
    axes = [axes]

for i, n_comp in enumerate(component_range):
    ax = axes[i]
    leverage = leverage_results.get(n_comp)
    if leverage is not None and not np.isnan(leverage).all():
        bars = ax.bar(sample_name, leverage)
        ax.set_ylabel("Leverage", fontsize=12)
        ax.set_title(f"Component = {n_comp}", fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=10) # y軸のラベルサイズも調整

        threshold = 2 * n_comp / tensor.shape[2]
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax.legend()

        for bar, name, lev_val in zip(bars, sample_name, leverage):
            if lev_val > threshold:
                ax.text(bar.get_x() + bar.get_width() / 2, lev_val, name,
                        ha='center', va='bottom', fontsize=10, color='blue', weight='bold')
    else:
        ax.set_title(f"Component = {n_comp} (Calculation Failed)")

plt.xlabel("Sample Name", fontsize=14)

# ▼▼▼ 各プロット間の余白をしっかり確保する命令を追加 ▼▼▼
# pad=3.0 で、各グラフの周囲に十分なスペースを確保します
fig.tight_layout(pad=3.0)

plt.show()

# %% [markdown]
# ## core consistency

# %%
# %% コアコンシステンシーの手動計算とプロット

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from tensorly.tenalg import multi_mode_dot

# --- 事前準備（これまでのステップで準備した変数） ---
# eem_array_normalized: 正規化済みでNaNを含むデータ配列
# -----------------------------------------------

# NaNを0で埋めた正規化済みデータ
eem_array_imputed = np.nan_to_num(eem_array_normalized, nan=0.0)
# テンソルに変換 (励起, 蛍光, サンプル)
tensor = tl.tensor(np.transpose(eem_array_imputed, (1, 2, 0)))

# 試行する成分数の範囲
# max_components = 9
component_range = range(1, max_components + 1)


print("--- コアコンシステンシーの計算を開始します (手動計算) ---")
core_consistencies = []

for n_comp in component_range:
    print(f"  - 成分数 = {n_comp} でモデルを構築中...")
    weights, factors = non_negative_parafac(tensor, rank=n_comp, n_iter_max=200,
                                            tol=1e-6, init='random')
    
    # --- コアコンシステンシーの手動計算 ---
    # 1. ローディング行列から、擬似逆行列を計算
    pseudo_inverses = [np.linalg.pinv(f) for f in factors]
    
    # 2. 現実のコアテンソル（G）を計算
    # G = X * (A^-1, B^-1, C^-1)
    core_tensor_G = multi_mode_dot(tensor, pseudo_inverses, modes=[0, 1, 2])

    # 3. 理想のコアテンソル（T）を作成（対角成分が1、他が0）
    ideal_core_T = tl.zeros_like(core_tensor_G)
    for i in range(n_comp):
        ideal_core_T[i, i, i] = 1
        
    # 4. コアテンソルGの全要素の二乗和 (ssq_G) を計算
    ssq_G = tl.sum(core_tensor_G**2)
    
    # 5. GとTの差の二乗和 (ssq_diff) を計算
    ssq_diff = tl.sum((core_tensor_G - ideal_core_T)**2)
    
    # 6. コアコンシステンシーを計算
    # 100 * (1 - (GとTの差の二乗和) / (Gの全要素の二乗和))
    if ssq_G > 1e-8: # ゼロ除算を避ける
        cc = (1 - (ssq_diff / ssq_G)) * 100
    else:
        cc = 0
        
    core_consistencies.append(cc)
    print(f"  - 成分数 = {n_comp} のコアコンシステンシー: {cc:.1f}%")

print("--- 計算が完了しました ---")

# --- 結果のプロット ---
plt.figure(figsize=(8, 5))
plt.plot(component_range, core_consistencies, 'o-', color='b', markersize=8)
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("Core Consistency (%)", fontsize=12)
plt.title("Core Consistency Diagnostic", fontsize=14)
plt.xticks(component_range)
plt.grid(True, linestyle='--', alpha=0.6)
# 60%のラインに補助線を追加
plt.axhline(y=60, color='r', linestyle='--', label='60% Threshold')
plt.legend()
plt.ylim(-5, 105) # 負の値も表示できるように調整
plt.show()

# %% [markdown]
# ## split-half

# %%
# %% スプリットハーフ分析

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- 事前準備（これまでのステップで準備した変数） ---
# eem_array_normalized: 正規化済みでNaNを含むデータ配列
# -----------------------------------------------

# NaNを0で埋めた正規化済みデータ
eem_array_imputed = np.nan_to_num(eem_array_normalized, nan=0.0)
# テンソルに変換 (励起, 蛍光, サンプル)
tensor = tl.tensor(np.transpose(eem_array_imputed, (1, 2, 0)))
num_samples = tensor.shape[2]


# 試行する成分数の範囲
# max_components = 9
component_range = range(2, max_components + 1)

print("--- スプリットハーフ分析を開始します ---")
similarity_scores = []

for n_comp in component_range:
    print(f"\n--- 成分数 = {n_comp} で分析中... ---")
    
    # 1. データセットをランダムに半分に分割
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    half1_indices = indices[:num_samples // 2]
    half2_indices = indices[num_samples // 2:]
    
    tensor_half1 = tl.tensor(tensor[:, :, half1_indices])
    tensor_half2 = tl.tensor(tensor[:, :, half2_indices])
    
    # 2. それぞれの半分でPARAFACを実行
    _, factors1 = non_negative_parafac(tensor_half1, rank=n_comp, n_iter_max=200, tol=1e-6, init='random')
    _, factors2 = non_negative_parafac(tensor_half2, rank=n_comp, n_iter_max=200, tol=1e-6, init='random')
    
    # 3. ローディングを比較
    excitation_loadings1, emission_loadings1, _ = factors1
    excitation_loadings2, emission_loadings2, _ = factors2
    
    # 類似度を計算 (最適なペアを見つけてマッチング)
    total_similarity = 0
    matched_indices2 = []
    
    for i in range(n_comp):
        best_match_idx = -1
        max_corr = -1
        for j in range(n_comp):
            if j in matched_indices2:
                continue
            corr_ex, _ = pearsonr(excitation_loadings1[:, i], excitation_loadings2[:, j])
            corr_em, _ = pearsonr(emission_loadings1[:, i], emission_loadings2[:, j])
            avg_corr = (corr_ex + corr_em) / 2
            
            if avg_corr > max_corr:
                max_corr = avg_corr
                best_match_idx = j
        
        total_similarity += max_corr
        matched_indices2.append(best_match_idx)
        
    avg_similarity = total_similarity / n_comp
    similarity_scores.append(avg_similarity * 100) # パーセントに変換
    print(f"  - 成分数 = {n_comp} の平均類似度: {avg_similarity*100:.1f}%")

print("--- 分析が完了しました ---")

# --- 結果のプロット ---
plt.figure(figsize=(8, 5))
plt.plot(component_range, similarity_scores, 'o-', color='g', markersize=8)
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("Split-Half Similarity (%)", fontsize=12)
plt.title("Split-Half Analysis", fontsize=14)
plt.xticks(component_range)
plt.grid(True, linestyle='--', alpha=0.6)
# 95%のラインに補助線を追加
plt.axhline(y=95, color='r', linestyle='--', label='95% Threshold')
plt.legend()
plt.ylim(0, 105)
plt.show()

# %% [markdown]
# ## ローディングの可視化

# %%
# %% 最終モデルの構築と結果の可視化

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import matplotlib.pyplot as plt

# --- 最適な成分数を設定 ---
# max_components = 9

# --- 事前準備 ---
# 以下の変数が事前に定義されている必要があります
# eem_array_normalized: 正規化済みの3次元データ配列 (サンプル x 励起 x 蛍光)
# sample_name: サンプル名のリスト (例: ['ABS', 'HDPE', ...])
# ex_bands_trimmed: 励起波長のリスト
# em_bands_trimmed: 蛍光波長のリスト

# NaNを0で埋めた正規化済みデータ
eem_array_imputed = np.nan_to_num(eem_array_normalized, nan=0.0)
# テンソルに変換 (励起, 蛍光, サンプル)
tensor = tl.tensor(np.transpose(eem_array_imputed, (1, 2, 0)))

# --- 最終モデルの計算 ---
print(f"--- 最適な成分数 = {max_components} で最終モデルを構築します ---")
weights, factors = non_negative_parafac(tensor, rank=max_components,
                                        n_iter_max=500, tol=1e-7, init='random')

# --- 結果の分解 ---
# factors は [励起ローディング, 蛍光ローディング, サンプルスコア]
excitation_loadings, emission_loadings, sample_scores = factors

# 各ローディングとスコアを最大値で正規化して見やすくする
excitation_loadings_norm = excitation_loadings / np.max(excitation_loadings, axis=0)
emission_loadings_norm = emission_loadings / np.max(emission_loadings, axis=0)
sample_scores_norm = sample_scores / np.max(sample_scores, axis=0)

print("--- モデル構築完了 ---")

# --- 結果の可視化 ---

# 1. 励起・蛍光ローディング（成分のスペクトル形状）のプロット
print("\n--- 抽出された成分のスペクトルをプロットします ---")
fig, axes = plt.subplots(max_components, 2, figsize=(12, 3 * max_components))
for i in range(max_components):
    # 励起ローディング
    axes[i, 0].plot(ex_bands_trimmed, excitation_loadings_norm[:, i])
    axes[i, 0].set_title(f'Component {i+1} - Excitation')
    axes[i, 0].set_xlabel('Excitation (nm)')
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 蛍光ローディング
    axes[i, 1].plot(em_bands_trimmed, emission_loadings_norm[:, i])
    axes[i, 1].set_title(f'Component {i+1} - Emission')
    axes[i, 1].set_xlabel('Emission (nm)')
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()
plt.show()



# %%
# %% 全成分のスペクトルを一枚の図にまとめてプロット

# --- 事前準備は変更なし ---
# (excitation_loadings_norm, emission_loadings_norm, sample_scores など)
# ---

print("\n--- 抽出された全成分のスペクトルを一枚の図にまとめてプロットします ---")

# --- 1. グラフ全体の準備 ---
# ループの外で、成分の数だけ縦に並ぶサブプロットを作成
fig, axes = plt.subplots(max_components, 1, figsize=(10, 4 * max_components))

# 成分が1つの場合でも、axesがリストになるように調整
if max_components == 1:
    axes = [axes]

# --- 2. 各成分についてループし、対応するサブプロットに描画 ---
for i in range(max_components):
    # 対応するサブプロット(ax1)を選択
    ax1 = axes[i]
    ax2 = ax1.twinx()  # 各サブプロットでY軸を共有

    # --- スペクトルのプロット ---
    color1 = 'tab:blue'
    ax1.plot(ex_bands_trimmed, excitation_loadings_norm[:, i], color=color1, lw=2.5)
    ax1.set_ylabel(f'Comp. {i+1} Ex (norm)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.6)

    color2 = 'tab:red'
    ax2.plot(em_bands_trimmed, emission_loadings_norm[:, i], color=color2, lw=2.5)
    ax2.set_ylabel(f'Comp. {i+1} Em (norm)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    # --- ピーク波長の検出と描画 ---
    ex_peak_idx = np.argmax(excitation_loadings_norm[:, i])
    ex_peak_wl = ex_bands_trimmed[ex_peak_idx]
    ax1.axvline(x=ex_peak_wl, color=color1, linestyle='--', alpha=0.8)
    ax1.text(ex_peak_wl - 3, ax1.get_ylim()[1] * 0.5, f'{ex_peak_wl:.0f} nm',
             color=color1, rotation='vertical', ha='right', va='center', fontsize=11)

    em_peak_idx = np.argmax(emission_loadings_norm[:, i])
    em_peak_wl = em_bands_trimmed[em_peak_idx]
    ax2.axvline(x=em_peak_wl, color=color2, linestyle='--', alpha=0.8)
    ax2.text(em_peak_wl + 3, ax2.get_ylim()[1] * 0.5, f'{em_peak_wl:.0f} nm',
             color=color2, rotation='vertical', ha='left', va='center', fontsize=11)
    
    # --- Y軸の範囲を0からに固定 ---
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

# --- 3. 仕上げ ---
# 最後のサブプロットにのみX軸のラベルを表示
axes[-1].set_xlabel('Wavelength (nm)', fontsize=12)

# 図全体のレイアウトを調整
fig.tight_layout()
plt.show()
    

# %%
# 2. サンプルスコア（各MPの成分含有量）のプロット
print("\n--- 各MPの成分含有量をプロットします ---")
plt.figure(figsize=(10, 6))
# ヒートマップで可視化
plt.imshow(sample_scores_norm.T, cmap='viridis', aspect='auto')
plt.yticks(ticks=np.arange(max_components), labels=[f'Component {i+1}' for i in range(max_components)])
plt.xticks(ticks=np.arange(len(sample_name)), labels=sample_name, rotation=45, ha='right')
plt.colorbar(label='Relative Concentration (normalized)')
plt.title('Component Scores for Each Microplastic', fontsize=14)
plt.show()

# %%
# %% 全成分のEEMを一枚の図にまとめてプロット

# --- 事前準備は変更なし ---
# (excitation_loadings, emission_loadingsなど)
# ---

print("\n--- 分離された全成分のEEMを一枚の図にまとめてプロットします ---")

# --- 1. グラフ全体の準備 ---
# サブプロットの行数と列数をいい感じに決める (例: 3列で表示)
n_cols = 3
n_rows = (max_components + n_cols - 1) // n_cols # 切り上げ除算
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
# axesを1次元のリストにして扱いやすくする
axes = axes.flatten()

# --- 2. 各成分についてループし、対応するサブプロットに描画 ---
for i in range(max_components):
    ax = axes[i] # 対応するサブプロットを選択

    # --- 外積を計算して、成分のEEMを再構成 ---
    component_eem = np.outer(excitation_loadings[:, i], emission_loadings[:, i])

    # --- ヒートマップとしてプロット ---
    im = ax.imshow(
        component_eem,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[em_bands_trimmed[0], em_bands_trimmed[-1], ex_bands_trimmed[0], ex_bands_trimmed[-1]]
    )

    # --- 仕上げ ---
    ax.set_title(f'Component {i+1}')
    ax.set_xlabel('Emission (nm)')
    ax.set_ylabel('Excitation (nm)')
    fig.colorbar(im, ax=ax)

# --- 3. 空のサブプロットを非表示にする ---
for j in range(max_components, len(axes)):
    axes[j].axis('off')

# --- 4. 図全体のレイアウトを調整 ---
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 残差の可視化

# %%
# %% 全サンプルの残差プロット（転置なし・ピーク表示付き）

import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

# --- 事前準備（最終モデルの計算結果から） ---
# tensor: 前処理済みの全サンプルEEMテンソル (励起, 蛍光, サンプル)
# weights: PARAFACの計算結果
# factors: PARAFACの計算結果 [励起ローディング, 蛍光ローディング, サンプルスコア]
# sample_name: サンプル名のリスト
# ex_bands_trimmed, em_bands_trimmed: 波長リスト
# ---------------------------------------------

# 1. PARAFACモデルによる再現テンソルを計算
reconstructed_tensor = tl.cp_to_tensor((weights, factors))

# 2. 残差テンソルを計算
residual_tensor = tensor - reconstructed_tensor

print("--- 全サンプルの元のEEMと残差EEM（転置なし）をプロットします ---")

# 3. 全てのサンプルについてループ処理
for i, sample in enumerate(sample_name):

    # 4. 対象サンプルの元のEEMと残差EEMを準備
    original_eem = tensor[:, :, i]
    residual_eem = residual_tensor[:, :, i]

    # 5. 残差EEMのピーク位置を特定
    peak_idx_flat = np.argmax(residual_eem)
    peak_ex_idx, peak_em_idx = np.unravel_index(peak_idx_flat, residual_eem.shape)
    
    peak_ex_wl = ex_bands_trimmed[peak_ex_idx]
    peak_em_wl = em_bands_trimmed[peak_em_idx]

    # 6. プロット作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Original vs. Residual EEM for: {sample}', fontsize=16)

    # --- 元のEEMプロット ---
    # ▼▼▼ .T を削除し、extentと軸ラベルを修正 ▼▼▼
    im1 = axes[0].imshow(original_eem, origin='lower', aspect='auto', cmap='viridis',
                         extent=[em_bands_trimmed[0], em_bands_trimmed[-1], ex_bands_trimmed[0], ex_bands_trimmed[-1]])
    axes[0].set_title('Original EEM')
    axes[0].set_xlabel('Emission (nm)')
    axes[0].set_ylabel('Excitation (nm)')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    fig.colorbar(im1, ax=axes[0], label='Intensity')

    # --- 残差EEMプロット ---
    # vminとvmaxを元のEEMと合わせてスケールを統一
    # ▼▼▼ .T を削除し、extentと軸ラベル、textの位置を修正 ▼▼▼
    im2 = axes[1].imshow(residual_eem, origin='lower', aspect='auto', cmap='viridis',
                         extent=[em_bands_trimmed[0], em_bands_trimmed[-1], ex_bands_trimmed[0], ex_bands_trimmed[-1]],
                         vmin=np.min(original_eem), vmax=np.max(original_eem))
    axes[1].set_title('Residual EEM (Model Error)')
    axes[1].set_xlabel('Emission (nm)')
    
    # # ピーク位置に十字線を描画
    # axes[1].axhline(y=peak_ex_wl, color='red', linestyle='--', alpha=0.7) # 水平線 (励起)
    # axes[1].axvline(x=peak_em_wl, color='red', linestyle='--', alpha=0.7) # 垂直線 (蛍光)
    
    # # 軸上に波長を表示
    # # Y軸（励起）
    # axes[1].text(axes[1].get_xlim()[0], peak_ex_wl, f'{peak_ex_wl:.0f} nm ',
    #            color='red', ha='right', va='center', weight='bold')
    # # X軸（蛍光）
    # axes[1].text(peak_em_wl, axes[1].get_ylim()[0], f'\n{peak_em_wl:.0f} nm',
    #            color='red', ha='center', va='top', weight='bold')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    fig.colorbar(im2, ax=axes[1], label='Intensity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [markdown]
# ---

# %% [markdown]
# 

# %% [markdown]
# 


