# %% [markdown]
# # labelmeの結果可視化

# %%
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path

# --- ユーザーが設定する項目 ---
# folder_name = "MPs_20250911"
folder_name = "MPs_20250905_2"

# プロジェクトのメインディレクトリとファイル名を指定
file_stem =f"{folder_name}_Ex-1_Em-1_ET300_step1"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）

json_path = main_dir / (file_stem + ".json")
image_path = main_dir / (file_stem + ".tiff")
#-------------------------------------------------------------------



# JSONファイルを読み込む
with open(json_path, 'r') as f:
    data = json.load(f)

# ラベル名と数値の対応辞書を作成
labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0

# 画像を読み込む
img = np.asarray(Image.open(image_path))

# マスクを生成
lbl, _ = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

# 各ラベルごとに画素値を抽出し、統計量を算出
results = {}
for label_name, value in label_name_to_value.items():
    if label_name == '_background_':
        continue
    
    # ラベルに対応するマスクを作成
    label_mask = (lbl == value)
    pixel_values = img[label_mask]
    
    if len(pixel_values) > 0:
        results[label_name] = {
            'pixel_count': len(pixel_values),
            'mean': np.mean(pixel_values),
            'std_dev': np.std(pixel_values),
            'max_value': np.max(pixel_values),
            'min_value': np.min(pixel_values)
        }
    else:
        results[label_name] = "No pixels found for this label."

# 結果を整形して表示
for label, stats in results.items():
    print(f'--- ラベル: {label} ---')
    if isinstance(stats, str):
        print(stats)
    else:
        print(f'画素数: {stats["pixel_count"]}')
        print(f'平均値: {stats["mean"]}')
        print(f'標準偏差: {stats["std_dev"]}')
        print(f'最大値: {stats["max_value"]}')
        print(f'最小値: {stats["min_value"]}')
    print('-------------------------')

# %%
import numpy as np
import labelme
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path

# # プロジェクトのメインディレクトリ
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")

# # 画像とJSONファイルのパス
# json_path = main_dir / "MPs_20250905_2_Ex-1_Em-1_ET300_step1.json"
# image_path = main_dir / "MPs_20250905_2_Ex-1_Em-1_ET300_step1.tiff"
# JSONファイルを読み込む
with open(json_path, 'r') as f:
    data = json.load(f)

print(image_path)

# 画像を読み込む
img = np.asarray(Image.open(image_path))

# label_name_to_value を作成
label_name_to_value = {"_background_": 0}  # 背景は0
for shape in data["shapes"]:
    label_name = shape["label"]
    if label_name not in label_name_to_value:
        label_name_to_value[label_name] = len(label_name_to_value)

# マスクを生成（新しいAPI）
lbl, _ = labelme.utils.shapes_to_label(
    img_shape=img.shape,
    shapes=data["shapes"],
    label_name_to_value=label_name_to_value
)

# マスクを可視化
mask = (lbl > 0)

plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
plt.imshow(mask, cmap='jet', alpha=0.5)
plt.title('Mask Visualization')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 画像ファイルのパスを指定してください
# 基準画像のパス
# reference_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2/MPs_20250905_2_Ex-1_Em-1_ET300_step1.tiff")
# 比較したい分光画像のパス（例）
spectral_image_path = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}/{folder_name}_Ex360_Em480_ET10000_step1.tiff")

# 画像を読み込み
try:
    ref_img = np.asarray(Image.open(image_path))
    spec_img = np.asarray(Image.open(spectral_image_path))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("ファイルパスを確認してください。")
    exit()

# 画像のサイズを確認
if ref_img.shape != spec_img.shape:
    print("Error: 画像のサイズが異なります。")
    exit()

# 画像を重ねて表示
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(ref_img, cmap='gray', alpha=1.0) # 基準画像を背景に表示
ax.imshow(spec_img, cmap='jet', alpha=0.5) # 分光画像を半透明で重ねて表示
ax.set_title('Image Alignment Check')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from skimage.registration import phase_cross_correlation

# # 画像ファイルのパスを指定
# reference_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.tiff")
# spectral_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex260_Em280_ET20000_step1.tiff")

# 画像を読み込み
try:
    ref_img = np.asarray(Image.open(image_path))
    spec_img = np.asarray(Image.open(spectral_image_path))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("ファイルパスを確認してください。")
    exit()

# 画像のサイズを確認
if ref_img.shape != spec_img.shape:
    print("Error: 画像のサイズが異なります。")
    exit()

# 位相相関法でずれを計算
# output: ずれの量 (y, x), 誤差, 位相相関のピーク
shift, error, phase = phase_cross_correlation(ref_img, spec_img)

# ずれの量（ピクセル単位）を表示
print(f'基準画像に対する分光画像のずれ:')
print(f'  y方向 (縦): {shift[0]:.2f} ピクセル')
print(f'  x方向 (横): {shift[1]:.2f} ピクセル')
print(f'  誤差: {error:.4f}')

# ずれが非常に小さい（0に近い）ことを確認し、問題ないと判断
if np.sqrt(shift[0]**2 + shift[1]**2) < 1.0:
    print("\n画像間のずれは1ピクセル未満です。位置合わせは正確であると考えられます。")
else:
    print("\n画像間にずれがある可能性があります。再撮影または画像補正を検討してください。")

# %% [markdown]
# # ピクセル単位での分類モデル

# %% [markdown]
# ## ラベリング結果の可視化

# %%
import labelme
import json
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # 凡例用
from matplotlib.colors import ListedColormap


# --- ユーザーが設定する項目 ---
folder_name = "MPs_20250911"
# folder_name = "MPs_20250905_2"

# プロジェクトのメインディレクトリとファイル名を指定
file_stem =f"{folder_name}_Ex-1_Em-1_ET300_step1"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）
#-------------------------------------------------------------------


json_path = main_dir / (reference_file_stem + ".json")
image_path = main_dir / (reference_file_stem + ".tiff") # ラベリングに使用した元の画像パス

print(f"Loading JSON file: {json_path}")
print(f"Loading original image: {image_path}")

try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSONファイル '{json_path}' が見つかりません。")
    exit()

try:
    original_img = np.asarray(Image.open(image_path))
except FileNotFoundError:
    print(f"Error: 元画像ファイル '{image_path}' が見つかりません。")
    exit()

# ラベル名と数値の対応付け
labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0 # 背景には0を割り当てる

# ラベルマップ（数値）を生成
lbl, _ = labelme.utils.shapes_to_label(original_img.shape, data['shapes'], label_name_to_value)

# 可視化用のカラーマップを生成
# ラベル数に応じて色を割り当てる
unique_labels = np.unique(lbl)
colors = plt.cm.get_cmap('tab10', len(unique_labels))
colored_mask = np.zeros((*lbl.shape, 3), dtype=np.uint8)
legend_patches = []
label_value_to_color = {}

for i, label_value in enumerate(unique_labels):
    if label_value == 0:
        color = np.array([0, 0, 0])
        label_name = 'background'
    else:
        # 修正: np.array()で一度配列に変換してからastype()を適用
        color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
        label_name = list(label_name_to_value.keys())[list(label_name_to_value.values()).index(label_value)]
        legend_patches.append(mpatches.Patch(color=color/255., label=label_name))
    
    colored_mask[lbl == label_value] = color

# 元画像とカラーマスクを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(colored_mask)
axes[1].set_title('Labeled Mask (Pixel-wise Labels)')
axes[1].axis('off')

plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

print("\n可視化されたラベリング結果を確認しました。")

# %% [markdown]
# ---
# # 2つのデータセットを用いた交差検証
# ## 学習・テスト対象：背景とプラスチックラベル域のピクセル

# %% [markdown]
# ### データセットの生成

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザーが設定する項目 ---
# main_dir や reference_file_stem はご自身の環境に合わせて設定してください
folder_name = "MPs_20250911"
# folder_name = "MPs_20250905_2"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名
# ------------------------------

# 1. 基準となるJSONファイルを読み込む
json_path = main_dir / (reference_file_stem + ".json")
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"エラー: JSONファイルが見つかりません。パスを確認してください: {json_path}")
    exit()

# 2. すべての分光画像の画素値をピクセル単位で抽出 (この部分は変更なし)
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
image_files = list(main_dir.glob("*.tiff"))
if not image_files:
    print("エラー: TIFF画像ファイルが見つかりません。")
    exit()

pixel_features_df = pd.DataFrame()
image_size = (data['imageHeight'], data['imageWidth'])
print(f"画像サイズを検出しました: {image_size}")

for image_path in image_files:
    if '-1_Em-1' in image_path.stem:
        continue
    match = wavelength_pattern.search(image_path.name)
    if not match:
        continue
    ex_wavelength = int(match.group(1))
    em_wavelength = int(match.group(2))
    if ex_wavelength == em_wavelength:
        continue

    try:
        img = np.asarray(Image.open(image_path))
        if img.shape[:2] != image_size:
            print(f"警告: {image_path.name} のサイズが異なります。スキップします。")
            continue
        pixel_features_df[f'Ex{ex_wavelength}_Em{em_wavelength}'] = img.flatten()
    except Exception as e:
        print(f"警告: {image_path.name} の処理に失敗しました。理由: {e}")
        continue

# (スクリプト前半は変更なし)
# ...

# 3. ★★★ ラベル情報の処理方法を修正 ★★★
print("\nスペクトルデータの抽出が完了しました。ラベル情報を結合・整形します...")

# 3-0. ★★★ 元の位置情報をインデックスとして保存 ★★★
pixel_features_df.reset_index(inplace=True)
pixel_features_df.rename(columns={'index': 'original_index'}, inplace=True)

# 3-1. JSON内のラベルにのみ数値を割り当て
labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
# (以下、pixel_label_maskの作成までは変更なし)
# ...
pixel_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)
pixel_labels_flat = pixel_label_mask.flatten()

# 3-2. ラベル名を対応付け
value_to_label_name = {v: k for k, v in label_name_to_value.items()}
value_to_label_name[0] = '_unlabeled_'
pixel_features_df['label_name'] = pd.Series(pixel_labels_flat).map(value_to_label_name)

# 3-3. 不要なラベルを持つピクセルを除外
labels_to_exclude = ['other', '_unlabeled_']
pixel_features_df = pixel_features_df[~pixel_features_df['label_name'].isin(labels_to_exclude)].copy()

# (以降の処理は変更なし)

# 3-4. 不要なラベルを持つピクセルを除外
initial_rows = len(pixel_features_df)
labels_to_exclude = ['other', '_unlabeled_']
pixel_features_df = pixel_features_df[~pixel_features_df['label_name'].isin(labels_to_exclude)].copy()

# 3-5. 'background_ref' を 'background' に名称変更
if 'background_ref' in pixel_features_df['label_name'].unique():
    pixel_features_df['label_name'] = pixel_features_df['label_name'].replace({'background_ref': 'background'})
    print("'background_ref' を 'background' に名称変更しました。")

final_rows = len(pixel_features_df)
print(f"除外対象 {labels_to_exclude} を持つピクセルを削除しました。")
print(f"処理後の総ピクセル数: {final_rows} (削除されたピクセル数: {initial_rows - final_rows})")

# 4. データセットをCSVとして保存
output_dir = main_dir / "csv"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = output_dir / "pixel_features_with_background.csv"
pixel_features_df.to_csv(output_csv_path, index=False)

print(f'\n新しいデータセットが作成されました: {output_csv_path}')
print('\nデータセットのプレビュー:')
print(pixel_features_df.head())
print('\n含まれるラベル一覧:')
print(pixel_features_df['label_name'].unique())

# %% [markdown]
# ----

# %% [markdown]
# ### t-SNE データセットの可視化

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- ユーザー設定 (ここだけ編集してください) ---
# ★ 解析したいデータフォルダ名を設定
folder_name = "MPs_20250911"
# ----------------------------------------------


# 1. パスの設定とCSVファイルの読み込み
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
input_csv_path = main_dir / "csv" / "pixel_features_with_background.csv"

print(f"データファイルを読み込みます: {input_csv_path}")
df = pd.read_csv(input_csv_path)
label_column = 'label_name'
if label_column not in df.columns:
    print(f"\n--- エラー ---")
    print(f"読み込んだCSVファイルに '{label_column}' 列が存在しません。")
    exit()

print("\nデータ読み込み成功。")

# 3. 各クラスから均等にデータをサンプリング
n_samples_per_class = 2000
print(f"\n各クラスから最大 {n_samples_per_class} 点をサンプリングします...")

sampled_dfs = []
for label in df[label_column].unique():
    group = df[df[label_column] == label]
    sample = group.sample(n=min(len(group), n_samples_per_class), random_state=42)
    sampled_dfs.append(sample)
sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"サンプリング後のデータセットサイズ: {len(sampled_df)} ピクセル")

# 4. データの前処理
labels = sampled_df[label_column]
columns_to_drop = [label_column]
if 'label_value' in sampled_df.columns:
    columns_to_drop.append('label_value')
features = sampled_df.drop(columns=columns_to_drop)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. t-SNEの計算
print("\nt-SNEの計算を開始します...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(features_scaled)
end_time = time.time()
print(f"t-SNE計算完了。実行時間: {end_time - start_time:.2f} 秒")

# 6. 結果の可視化
print("\n結果をプロットします...")
df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
df_tsne['label'] = labels

# 6-1. ユニークなラベル名を取得し、ソート
unique_labels = sorted(df_tsne['label'].unique())

# 6-2. カラーパレットを準備
plastic_colors = plt.cm.get_cmap('tab10', len(unique_labels))

# 6-3. ラベル名と色を対応付ける辞書を作成
color_map = {}
plastic_color_index = 0
for label in unique_labels:
    # ★★★ 変更点2: チェックする名前を'background'に変更 ★★★
    if label == 'background':
        color_map[label] = 'lightgray'
    else:
        color_map[label] = plastic_colors(plastic_color_index)
        plastic_color_index += 1

# 6-4. プロットの実行
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    hue_order=unique_labels,
    palette=color_map,
    data=df_tsne,
    legend="full",
    # ★★★ ここを変更 ★★★
    alpha=1.0,  # 点を不透明に変更 (0.7 -> 1.0)
    s=50        # 点のサイズを大きく変更 (20 -> 50)
)

plt.title(f't-SNE Plot of Spectral Data (Sampled from {folder_name})', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- ユーザー設定 (ここだけ編集してください) ---
# ★ 解析したいデータフォルダ名を設定
folder_name = "MPs_20250905_2"
# ----------------------------------------------


# 1. パスの設定とCSVファイルの読み込み
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
input_csv_path = main_dir / "csv" / "pixel_features_with_background.csv"

print(f"データファイルを読み込みます: {input_csv_path}")
df = pd.read_csv(input_csv_path)
label_column = 'label_name'
if label_column not in df.columns:
    print(f"\n--- エラー ---")
    print(f"読み込んだCSVファイルに '{label_column}' 列が存在しません。")
    exit()

print("\nデータ読み込み成功。")

# 3. 各クラスから均等にデータをサンプリング
n_samples_per_class = 2000
print(f"\n各クラスから最大 {n_samples_per_class} 点をサンプリングします...")

sampled_dfs = []
for label in df[label_column].unique():
    group = df[df[label_column] == label]
    sample = group.sample(n=min(len(group), n_samples_per_class), random_state=42)
    sampled_dfs.append(sample)
sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"サンプリング後のデータセットサイズ: {len(sampled_df)} ピクセル")

# 4. データの前処理
labels = sampled_df[label_column]
columns_to_drop = [label_column]
if 'label_value' in sampled_df.columns:
    columns_to_drop.append('label_value')
features = sampled_df.drop(columns=columns_to_drop)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. t-SNEの計算
print("\nt-SNEの計算を開始します...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(features_scaled)
end_time = time.time()
print(f"t-SNE計算完了。実行時間: {end_time - start_time:.2f} 秒")

# 6. 結果の可視化
print("\n結果をプロットします...")
df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
df_tsne['label'] = labels

# ★★★ 変更点1: プロット用に'_background_'を'background'に名称変更 ★★★
df_tsne['label'] = df_tsne['label'].replace({'_background_': 'background'})

# 6-1. ユニークなラベル名を取得し、ソート
unique_labels = sorted(df_tsne['label'].unique())

# 6-2. カラーパレットを準備
plastic_colors = plt.cm.get_cmap('tab10', len(unique_labels))

# 6-3. ラベル名と色を対応付ける辞書を作成
color_map = {}
plastic_color_index = 0
for label in unique_labels:
    # ★★★ 変更点2: チェックする名前を'background'に変更 ★★★
    if label == 'background':
        color_map[label] = 'lightgray'
    else:
        color_map[label] = plastic_colors(plastic_color_index)
        plastic_color_index += 1

# 6-4. プロットの実行
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    hue_order=unique_labels,
    palette=color_map,
    data=df_tsne,
    legend="full",
    # ★★★ ここを変更 ★★★
    alpha=1.0,  # 点を不透明に変更 (0.7 -> 1.0)
    s=50        # 点のサイズを大きく変更 (20 -> 50)
)

plt.title(f't-SNE Plot of Spectral Data (Sampled from {folder_name})', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %% [markdown]
# ----
# 

# %% [markdown]
# ### データセットの学習

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# ------------------------------

# データセットのパスを定義
dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"
dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_with_background.csv"

# データセット1を読み込む
try:
    df1 = pd.read_csv(dataset1_csv_path)
    print(f"データセット1を正常に読み込みました: {dataset1_csv_path}")
except FileNotFoundError:
    print(f"エラー: {dataset1_csv_path} が見つかりません。")
    exit()

# データセット2を読み込む
try:
    df2 = pd.read_csv(dataset2_csv_path)
    print(f"データセット2を正常に読み込みました: {dataset2_csv_path}")
except FileNotFoundError:
    print(f"エラー: {dataset2_csv_path} が見つかりません。")
    exit()

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

# original_indexを除外する、正しいprepare_data関数
def prepare_data(df):
    """データフレームから特徴量とラベルを分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    
    # 除外する列のリストを作成
    columns_to_drop = ['label_name']
    if 'original_index' in df.columns:
        columns_to_drop.append('original_index')
    if 'label_value' in df.columns:
        columns_to_drop.append('label_value')
    
    X = df.drop(columns=columns_to_drop)
    y = df['label_name']
    return X, y

def train_and_save(X_train, y_train, model_save_path):
    print(f"\n--- モデルの学習を開始: {model_save_path.name} ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, model_save_path)
    print("学習済みモデルを保存しました。")

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# --------------------

try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv")
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / "pixel_features_with_background.csv")
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません。パスを確認してください。: {e.filename}")
    exit()

# データの前処理
X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# モデル1を学習・保存
train_and_save(X1, y1, main_dir / "model_trained_on_dataset1.joblib")
print(f"データセット1（{dataset1_folder_name}）学習完了")

# モデル2を学習・保存
train_and_save(X2, y2, main_dir / "model_trained_on_dataset2.joblib")
print(f"データセット2（{dataset2_folder_name}）学習完了")


# %% [markdown]
# ### 分類モデルの交差検証

# %%
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import time

# ★★★ ここを修正 ★★★
def prepare_data(df):
    """データフレームから特徴量とラベルを分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    
    # 除外する列のリストを作成
    columns_to_drop = ['label_name']
    # 'original_index' も特徴量ではないため、存在すれば除外
    if 'original_index' in df.columns:
        columns_to_drop.append('original_index')
    # 'label_value' はもう存在しないが、念のためチェック
    if 'label_value' in df.columns:
        columns_to_drop.append('label_value')
    
    X = df.drop(columns=columns_to_drop)
    y = df['label_name']
    return X, y
# ★★★ ここまで ★★★

def evaluate_model(model_path, X_test, y_test, model_name):
    print(f"\n--- {model_name}の評価を開始 ---")
    start_time = time.time()
    
    # モデルの読み込み
    model = joblib.load(model_path)
    
    # 評価
    y_pred = model.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print('--- 精度評価レポート ---')
    print(classification_report(y_test, y_pred))
    print(f"評価が完了しました。実行時間: {elapsed_time:.2f}秒")
    
    feature_importances = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    return importance_df, model

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# --------------------

try:
    dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"
    dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_with_background.csv"
    
    df1 = pd.read_csv(dataset1_csv_path)
    df2 = pd.read_csv(dataset2_csv_path)
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません。パスを確認してください。: {e.filename}")
    exit()

# データの前処理
X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# 交差検証1: モデル1を評価
importance1, model1 = evaluate_model(main_dir / "model_trained_on_dataset1.joblib", X2, y2, "モデル1 (Dataset1で学習)")

# 交差検証2: モデル2を評価
importance2, model2 = evaluate_model(main_dir / "model_trained_on_dataset2.joblib", X1, y1, "モデル2 (Dataset2で学習)")

# --- 結果を保存するコードブロック ---
data1_output_dir = main_dir / dataset1_folder_name
data2_output_dir = main_dir / dataset2_folder_name

# 交差検証1の結果
with open(data1_output_dir / "classification_report_model1.txt", "w") as f:
    f.write(classification_report(y2, model1.predict(X2)))

# 交差検証2の結果
with open(data2_output_dir / "classification_report_model2.txt", "w") as f:
    f.write(classification_report(y1, model2.predict(X1)))

# 特徴量重要度をCSVファイルとして保存
importance1.to_csv(data1_output_dir / "csv" / "importance_from_dataset1.csv", index=False)
importance2.to_csv(data2_output_dir / "csv" / "importance_from_dataset2.csv", index=False)

print("\n--- 全ての処理が完了しました ---")
print(f"精度レポートと重要度ランキングは各データセットフォルダに保存されました。")


# %% [markdown]
# ### 分類結果の可視化

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import labelme
import json

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセット（正解ラベルを表示する側）
test_dataset_folder_name = "MPs_20250905_2" 
# 使用する学習済みモデル
model_path = main_dir / "model_trained_on_dataset1.joblib"
# --------------------

# 1. データの準備
print("--- データの準備 ---")

# ★★★ 色とラベルの対応をここで一元管理 ★★★
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime", "background": "gray",
    "background_ref": "gray" # 念のため_refも同じ色に
}

# 評価対象のJSONとCSVファイルのパスを定義
json_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.json"
test_csv_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_with_background.csv"

# 2. 正解ラベルマスクの作成 (左側の画像)
print("正解ラベルマスクを作成しています...")
with open(json_path, 'r') as f:
    json_data = json.load(f)

img_height = json_data['imageHeight']
img_width = json_data['imageWidth']

# labelme形式で数値マスクを生成
labels_in_json = sorted(list(set(shape['label'] for shape in json_data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
numeric_ground_truth_mask, _ = labelme.utils.shapes_to_label((img_height, img_width), json_data['shapes'], label_name_to_value)

# 数値マスクをカラーマスクに変換
ground_truth_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
for label_name, label_value in label_name_to_value.items():
    # 'background_ref' を 'background' として扱う
    display_label = 'background' if label_name == 'background_ref' else label_name
    
    if display_label in label_to_color_map:
        color_rgb = (np.array(plt.cm.colors.to_rgb(label_to_color_map[display_label])) * 255).astype(np.uint8)
        ground_truth_mask[numeric_ground_truth_mask == label_value] = color_rgb

# 3. 予測結果マスクの作成 (右側の画像)
print("モデルによる予測とマスク作成を実行しています...")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

df_test = pd.read_csv(test_csv_path)

original_indices = df_test['original_index'].values
X_test = df_test[feature_names_from_model]
y_pred = model.predict(X_test)

predicted_mask_flat = np.zeros((img_height * img_width, 3), dtype=np.uint8)
for label_name, color_name in label_to_color_map.items():
    pred_indices_in_y = np.where(y_pred == label_name)[0]
    if len(pred_indices_in_y) > 0:
        img_indices_to_paint = original_indices[pred_indices_in_y]
        color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
        predicted_mask_flat[img_indices_to_paint] = color_rgb

predicted_mask = predicted_mask_flat.reshape(img_height, img_width, 3)

# 4. 2つの画像を並べてプロット
print("\n--- 結果の比較表示 ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左: 正解ラベル
axes[0].imshow(ground_truth_mask)
axes[0].set_title('Ground Truth (Annotation)', fontsize=16)
axes[0].axis('off')

# 右: モデルの予測結果
axes[1].imshow(predicted_mask)
axes[1].set_title('Model Prediction', fontsize=16)
axes[1].axis('off')

# 共通の凡例を作成
legend_patches = []
# backgroundを除いた凡例を作成
plot_labels = {k: v for k, v in label_to_color_map.items() if k not in ['background_ref']}

for label_name, color in sorted(plot_labels.items()):
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
fig.legend(handles=legend_patches, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import labelme
import json

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセット（正解ラベルを表示する側）
test_dataset_folder_name = "MPs_20250911" 
# 使用する学習済みモデル
model_path = main_dir / "model_trained_on_dataset2.joblib"
# --------------------

# 1. データの準備
print("--- データの準備 ---")

# ★★★ 色とラベルの対応をここで一元管理 ★★★
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime", "background": "gray",
    "background_ref": "gray" # 念のため_refも同じ色に
}

# 評価対象のJSONとCSVファイルのパスを定義
json_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.json"
test_csv_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_with_background.csv"

# 2. 正解ラベルマスクの作成 (左側の画像)
print("正解ラベルマスクを作成しています...")
with open(json_path, 'r') as f:
    json_data = json.load(f)

img_height = json_data['imageHeight']
img_width = json_data['imageWidth']

# labelme形式で数値マスクを生成
labels_in_json = sorted(list(set(shape['label'] for shape in json_data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
numeric_ground_truth_mask, _ = labelme.utils.shapes_to_label((img_height, img_width), json_data['shapes'], label_name_to_value)

# 数値マスクをカラーマスクに変換
ground_truth_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
for label_name, label_value in label_name_to_value.items():
    # 'background_ref' を 'background' として扱う
    display_label = 'background' if label_name == 'background_ref' else label_name
    
    if display_label in label_to_color_map:
        color_rgb = (np.array(plt.cm.colors.to_rgb(label_to_color_map[display_label])) * 255).astype(np.uint8)
        ground_truth_mask[numeric_ground_truth_mask == label_value] = color_rgb

# 3. 予測結果マスクの作成 (右側の画像)
print("モデルによる予測とマスク作成を実行しています...")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

df_test = pd.read_csv(test_csv_path)

original_indices = df_test['original_index'].values
X_test = df_test[feature_names_from_model]
y_pred = model.predict(X_test)

predicted_mask_flat = np.zeros((img_height * img_width, 3), dtype=np.uint8)
for label_name, color_name in label_to_color_map.items():
    pred_indices_in_y = np.where(y_pred == label_name)[0]
    if len(pred_indices_in_y) > 0:
        img_indices_to_paint = original_indices[pred_indices_in_y]
        color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
        predicted_mask_flat[img_indices_to_paint] = color_rgb

predicted_mask = predicted_mask_flat.reshape(img_height, img_width, 3)

# 4. 2つの画像を並べてプロット
print("\n--- 結果の比較表示 ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左: 正解ラベル
axes[0].imshow(ground_truth_mask)
axes[0].set_title('Ground Truth (Annotation)', fontsize=16)
axes[0].axis('off')

# 右: モデルの予測結果
axes[1].imshow(predicted_mask)
axes[1].set_title('Model Prediction', fontsize=16)
axes[1].axis('off')

# 共通の凡例を作成
legend_patches = []
# backgroundを除いた凡例を作成
plot_labels = {k: v for k, v in label_to_color_map.items() if k not in ['background_ref']}

for label_name, color in sorted(plot_labels.items()):
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
fig.legend(handles=legend_patches, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %% [markdown]
# ---

# %% [markdown]
# ## 学習・テスト対象：プラスチックのピクセル域のみ

# %% [markdown]
# ### データセットの作成

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザーが設定する項目 ---
# main_dir や reference_file_stem はご自身の環境に合わせて設定してください
# folder_name = "MPs_20250911"
folder_name = "MPs_20250905_2"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名
# ------------------------------

# 1. 基準となるJSONファイルを読み込む
json_path = main_dir / (reference_file_stem + ".json")
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"エラー: JSONファイルが見つかりません。パスを確認してください: {json_path}")
    exit()

# 2. すべての分光画像の画素値をピクセル単位で抽出
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
image_files = list(main_dir.glob("*.tiff"))
if not image_files:
    print("エラー: TIFF画像ファイルが見つかりません。")
    exit()

pixel_features_df = pd.DataFrame()
image_size = (data['imageHeight'], data['imageWidth'])
print(f"画像サイズを検出しました: {image_size}")

for image_path in image_files:
    if '-1_Em-1' in image_path.stem:
        continue
    match = wavelength_pattern.search(image_path.name)
    if not match:
        continue
    ex_wavelength = int(match.group(1))
    em_wavelength = int(match.group(2))
    if ex_wavelength == em_wavelength:
        continue

    try:
        img = np.asarray(Image.open(image_path))
        if img.shape[:2] != image_size:
            print(f"警告: {image_path.name} のサイズが異なります。スキップします。")
            continue
        pixel_features_df[f'Ex{ex_wavelength}_Em{em_wavelength}'] = img.flatten()
    except Exception as e:
        print(f"警告: {image_path.name} の処理に失敗しました。理由: {e}")
        continue

# 3. ラベル情報の結合と整形
print("\nスペクトルデータの抽出が完了しました。ラベル情報を結合・整形します...")
pixel_features_df.reset_index(inplace=True)
pixel_features_df.rename(columns={'index': 'original_index'}, inplace=True)

labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}

pixel_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)
pixel_labels_flat = pixel_label_mask.flatten()

value_to_label_name = {v: k for k, v in label_name_to_value.items()}
value_to_label_name[0] = '_unlabeled_'
pixel_features_df['label_name'] = pd.Series(pixel_labels_flat).map(value_to_label_name)

# ★★★ ここからが変更点 ★★★
# 3-1. 'background_ref' を 'background' に名称変更 (除外処理の前に行う)
if 'background_ref' in pixel_features_df['label_name'].unique():
    pixel_features_df['label_name'] = pixel_features_df['label_name'].replace({'background_ref': 'background'})
    print("'background_ref' を 'background' に名称変更しました。")

# 3-2. 不要なラベルを持つピクセルを除外 (backgroundも除外対象に追加)
initial_rows = len(pixel_features_df)
labels_to_exclude = ['other', '_unlabeled_', 'background']
pixel_features_df = pixel_features_df[~pixel_features_df['label_name'].isin(labels_to_exclude)].copy()

final_rows = len(pixel_features_df)
print(f"除外対象 {labels_to_exclude} を持つピクセルを削除しました。")
print(f"処理後の総ピクセル数: {final_rows} (削除されたピクセル数: {initial_rows - final_rows})")
# ★★★ ここまでが変更点 ★★★

# 4. データセットをCSVとして保存
output_dir = main_dir / "csv"
output_dir.mkdir(parents=True, exist_ok=True)
# ★★★ 出力ファイル名を変更 ★★★
output_csv_path = output_dir / "pixel_features_plastics_only.csv"
pixel_features_df.to_csv(output_csv_path, index=False)

print(f'\n新しい「プラスチックのみ」のデータセットが作成されました: {output_csv_path}')
print('\nデータセットのプレビュー:')
print(pixel_features_df.head())
print('\n含まれるラベル一覧:')
print(pixel_features_df['label_name'].unique())


# %% [markdown]
# ### t-SNE

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- ユーザー設定 (ここだけ編集してください) ---
# ★ 解析したいデータフォルダ名を設定
folder_name = "MPs_20250911"
# ----------------------------------------------


# 1. パスの設定とCSVファイルの読み込み
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
input_csv_path = main_dir / "csv" / "pixel_features_plastics_only.csv"

print(f"データファイルを読み込みます: {input_csv_path}")
df = pd.read_csv(input_csv_path)
label_column = 'label_name'
if label_column not in df.columns:
    print(f"\n--- エラー ---")
    print(f"読み込んだCSVファイルに '{label_column}' 列が存在しません。")
    exit()

print("\nデータ読み込み成功。")

# 3. 各クラスから均等にデータをサンプリング
n_samples_per_class = 2000
print(f"\n各クラスから最大 {n_samples_per_class} 点をサンプリングします...")

sampled_dfs = []
for label in df[label_column].unique():
    group = df[df[label_column] == label]
    sample = group.sample(n=min(len(group), n_samples_per_class), random_state=42)
    sampled_dfs.append(sample)
sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"サンプリング後のデータセットサイズ: {len(sampled_df)} ピクセル")

# 4. データの前処理
labels = sampled_df[label_column]
columns_to_drop = [label_column]
if 'label_value' in sampled_df.columns:
    columns_to_drop.append('label_value')
features = sampled_df.drop(columns=columns_to_drop)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. t-SNEの計算
print("\nt-SNEの計算を開始します...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(features_scaled)
end_time = time.time()
print(f"t-SNE計算完了。実行時間: {end_time - start_time:.2f} 秒")

# 6. 結果の可視化
print("\n結果をプロットします...")
df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
df_tsne['label'] = labels

# 6-1. ユニークなラベル名を取得し、ソート
unique_labels = sorted(df_tsne['label'].unique())

# 6-2. カラーパレットを準備
plastic_colors = plt.cm.get_cmap('tab10', len(unique_labels))

# 6-3. ラベル名と色を対応付ける辞書を作成
color_map = {}
plastic_color_index = 0
for label in unique_labels:
    # ★★★ 変更点2: チェックする名前を'background'に変更 ★★★
    if label == 'background':
        color_map[label] = 'lightgray'
    else:
        color_map[label] = plastic_colors(plastic_color_index)
        plastic_color_index += 1

# 6-4. プロットの実行
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    hue_order=unique_labels,
    palette=color_map,
    data=df_tsne,
    legend="full",
    # ★★★ ここを変更 ★★★
    alpha=1.0,  # 点を不透明に変更 (0.7 -> 1.0)
    s=50        # 点のサイズを大きく変更 (20 -> 50)
)

plt.title(f't-SNE Plot of Spectral Data (Sampled from {folder_name})', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- ユーザー設定 (ここだけ編集してください) ---
# ★ 解析したいデータフォルダ名を設定
folder_name = "MPs_20250905_2"
# ----------------------------------------------


# 1. パスの設定とCSVファイルの読み込み
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
input_csv_path = main_dir / "csv" / "pixel_features_plastics_only.csv"

print(f"データファイルを読み込みます: {input_csv_path}")
df = pd.read_csv(input_csv_path)
label_column = 'label_name'
if label_column not in df.columns:
    print(f"\n--- エラー ---")
    print(f"読み込んだCSVファイルに '{label_column}' 列が存在しません。")
    exit()

print("\nデータ読み込み成功。")

# 3. 各クラスから均等にデータをサンプリング
n_samples_per_class = 2000
print(f"\n各クラスから最大 {n_samples_per_class} 点をサンプリングします...")

sampled_dfs = []
for label in df[label_column].unique():
    group = df[df[label_column] == label]
    sample = group.sample(n=min(len(group), n_samples_per_class), random_state=42)
    sampled_dfs.append(sample)
sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"サンプリング後のデータセットサイズ: {len(sampled_df)} ピクセル")

# 4. データの前処理
labels = sampled_df[label_column]
columns_to_drop = [label_column]
if 'label_value' in sampled_df.columns:
    columns_to_drop.append('label_value')
features = sampled_df.drop(columns=columns_to_drop)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. t-SNEの計算
print("\nt-SNEの計算を開始します...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(features_scaled)
end_time = time.time()
print(f"t-SNE計算完了。実行時間: {end_time - start_time:.2f} 秒")

# 6. 結果の可視化
print("\n結果をプロットします...")
df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
df_tsne['label'] = labels

# 6-1. ユニークなラベル名を取得し、ソート
unique_labels = sorted(df_tsne['label'].unique())

# 6-2. カラーパレットを準備
plastic_colors = plt.cm.get_cmap('tab10', len(unique_labels))

# 6-3. ラベル名と色を対応付ける辞書を作成
color_map = {}
plastic_color_index = 0
for label in unique_labels:
    # ★★★ 変更点2: チェックする名前を'background'に変更 ★★★
    if label == 'background':
        color_map[label] = 'lightgray'
    else:
        color_map[label] = plastic_colors(plastic_color_index)
        plastic_color_index += 1

# 6-4. プロットの実行
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    hue_order=unique_labels,
    palette=color_map,
    data=df_tsne,
    legend="full",
    # ★★★ ここを変更 ★★★
    alpha=1.0,  # 点を不透明に変更 (0.7 -> 1.0)
    s=50        # 点のサイズを大きく変更 (20 -> 50)
)

plt.title(f't-SNE Plot of Spectral Data (Sampled from {folder_name})', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ### データセットの学習

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# ------------------------------

# データセットのパスを定義
dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_plastics_only.csv"
dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_plastics_only.csv"

# データセット1を読み込む
try:
    df1 = pd.read_csv(dataset1_csv_path)
    print(f"データセット1を正常に読み込みました: {dataset1_csv_path}")
except FileNotFoundError:
    print(f"エラー: {dataset1_csv_path} が見つかりません。")
    exit()

# データセット2を読み込む
try:
    df2 = pd.read_csv(dataset2_csv_path)
    print(f"データセット2を正常に読み込みました: {dataset2_csv_path}")
except FileNotFoundError:
    print(f"エラー: {dataset2_csv_path} が見つかりません。")
    exit()

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

# original_indexを除外する、正しいprepare_data関数
def prepare_data(df):
    """データフレームから特徴量とラベルを分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    
    # 除外する列のリストを作成
    columns_to_drop = ['label_name']
    if 'original_index' in df.columns:
        columns_to_drop.append('original_index')
    if 'label_value' in df.columns:
        columns_to_drop.append('label_value')
    
    X = df.drop(columns=columns_to_drop)
    y = df['label_name']
    return X, y

def train_and_save(X_train, y_train, model_save_path):
    print(f"\n--- モデルの学習を開始: {model_save_path.name} ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, model_save_path)
    print("学習済みモデルを保存しました。")

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# --------------------

try:
    # ★★★ 読み込むCSVファイル名を変更 ★★★
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / "pixel_features_plastics_only.csv")
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / "pixel_features_plastics_only.csv")
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません。パスを確認してください。: {e.filename}")
    print("先に「データセット作成（プラスチックのみ）」スクリプトを実行してください。")
    exit()

# データの前処理
X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# ★★★ 保存するモデルファイル名を変更 ★★★
# モデル1を学習・保存
train_and_save(X1, y1, main_dir / "model_plastics_only_dataset1.joblib")
print(f"データセット1（{dataset1_folder_name}）学習完了")

# モデル2を学習・保存
train_and_save(X2, y2, main_dir / "model_plastics_only_dataset2.joblib")
print(f"データセット2（{dataset2_folder_name}）学習完了")


# %% [markdown]
# ### 分類モデルの交差検証・特徴量重要度

# %%
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import time

# original_indexを除外する、正しいprepare_data関数
def prepare_data(df):
    """データフレームから特徴量とラベルを分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    
    # 除外する列のリストを作成
    columns_to_drop = ['label_name']
    if 'original_index' in df.columns:
        columns_to_drop.append('original_index')
    if 'label_value' in df.columns:
        columns_to_drop.append('label_value')
    
    X = df.drop(columns=columns_to_drop)
    y = df['label_name']
    return X, y

def evaluate_model(model_path, X_test, y_test, model_name):
    print(f"\n--- {model_name}の評価を開始 ---")
    start_time = time.time()
    
    # モデルの読み込み
    model = joblib.load(model_path)
    
    # 評価
    y_pred = model.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print('--- 精度評価レポート ---')
    print(classification_report(y_test, y_pred))
    print(f"評価が完了しました。実行時間: {elapsed_time:.2f}秒")
    
    feature_importances = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    return importance_df, model

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# --------------------

try:
    # ★★★ 読み込むCSVファイル名を変更 ★★★
    dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_plastics_only.csv"
    dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_plastics_only.csv"
    
    df1 = pd.read_csv(dataset1_csv_path)
    df2 = pd.read_csv(dataset2_csv_path)
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません。パスを確認してください。: {e.filename}")
    exit()

# データの前処理
X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# ★★★ 読み込むモデルファイル名を変更 ★★★
# 交差検証1: モデル1を評価
importance1, model1 = evaluate_model(main_dir / "model_plastics_only_dataset1.joblib", X2, y2, "モデル1 (Dataset1で学習)")

# 交差検証2: モデル2を評価
importance2, model2 = evaluate_model(main_dir / "model_plastics_only_dataset2.joblib", X1, y1, "モデル2 (Dataset2で学習)")

# --- 結果を保存するコードブロック ---
data1_output_dir = main_dir / dataset1_folder_name
data2_output_dir = main_dir / dataset2_folder_name

# ★★★ 保存するレポートファイル名を変更 ★★★
# 交差検証1の結果
with open(data1_output_dir / "classification_report_plastics_only_model1.txt", "w") as f:
    f.write(classification_report(y2, model1.predict(X2)))

# 交差検証2の結果
with open(data2_output_dir / "classification_report_plastics_only_model2.txt", "w") as f:
    f.write(classification_report(y1, model2.predict(X1)))

# ★★★ 保存する重要度ファイル名を変更 ★★★
# 特徴量重要度をCSVファイルとして保存
importance1.to_csv(data1_output_dir / "csv" / "importance_plastics_only_from_dataset1.csv", index=False)
importance2.to_csv(data2_output_dir / "csv" / "importance_plastics_only_from_dataset2.csv", index=False)

print("\n--- 全ての処理が完了しました ---")
print(f"精度レポートと重要度ランキングは各データセットフォルダに保存されました。")


# %% [markdown]
# ### 分類結果の可視化

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import labelme
import json

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセット（正解ラベルを表示する側）
test_dataset_folder_name = "MPs_20250905_2" 
# 使用する学習済みモデル (プラスチックのみで学習させたモデル)
model_path = main_dir / "model_plastics_only_dataset1.joblib"
# --------------------

# 1. データの準備
print("--- データの準備 ---")

# ★★★ プラスチックの色のみを定義 ★★★
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime"
}

# 評価対象のJSONとCSVファイルのパスを定義
json_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.json"
# ★★★ プラスチックのみのCSVを読み込む ★★★
test_csv_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_plastics_only.csv"

# 2. 正解ラベルマスクの作成 (左側の画像)
print("正解ラベルマスク（プラスチックのみ）を作成しています...")
with open(json_path, 'r') as f:
    json_data = json.load(f)

img_height = json_data['imageHeight']
img_width = json_data['imageWidth']

# labelme形式で数値マスクを生成
labels_in_json = sorted(list(set(shape['label'] for shape in json_data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
numeric_ground_truth_mask, _ = labelme.utils.shapes_to_label((img_height, img_width), json_data['shapes'], label_name_to_value)

# 数値マスクをカラーマスクに変換 (★★★ 背景は描画しない ★★★)
ground_truth_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
for label_name, label_value in label_name_to_value.items():
    # プラスチックラベルのみを色付けする
    if label_name in label_to_color_map:
        color_rgb = (np.array(plt.cm.colors.to_rgb(label_to_color_map[label_name])) * 255).astype(np.uint8)
        ground_truth_mask[numeric_ground_truth_mask == label_value] = color_rgb

# 3. 予測結果マスクの作成 (右側の画像)
print("モデルによる予測とマスク作成を実行しています...")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

df_test = pd.read_csv(test_csv_path)

original_indices = df_test['original_index'].values
X_test = df_test[feature_names_from_model]
y_pred = model.predict(X_test)

predicted_mask_flat = np.zeros((img_height * img_width, 3), dtype=np.uint8)
for label_name, color_name in label_to_color_map.items():
    pred_indices_in_y = np.where(y_pred == label_name)[0]
    if len(pred_indices_in_y) > 0:
        img_indices_to_paint = original_indices[pred_indices_in_y]
        color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
        predicted_mask_flat[img_indices_to_paint] = color_rgb

predicted_mask = predicted_mask_flat.reshape(img_height, img_width, 3)

# 4. 2つの画像を並べてプロット
print("\n--- 結果の比較表示 ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左: 正解ラベル
axes[0].imshow(ground_truth_mask)
axes[0].set_title('Ground Truth (Plastics Only)', fontsize=16)
axes[0].axis('off')

# 右: モデルの予測結果
axes[1].imshow(predicted_mask)
axes[1].set_title('Model Prediction (Plastics Only)', fontsize=16)
axes[1].axis('off')

# 共通の凡例を作成
legend_patches = []
for label_name, color in sorted(label_to_color_map.items()):
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
fig.legend(handles=legend_patches, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import labelme
import json

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセット（正解ラベルを表示する側）
test_dataset_folder_name = "MPs_20250911" 
# 使用する学習済みモデル (プラスチックのみで学習させたモデル)
model_path = main_dir / "model_plastics_only_dataset2.joblib"
# --------------------

# 1. データの準備
print("--- データの準備 ---")

# ★★★ プラスチックの色のみを定義 ★★★
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime"
}

# 評価対象のJSONとCSVファイルのパスを定義
json_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.json"
# ★★★ プラスチックのみのCSVを読み込む ★★★
test_csv_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_plastics_only.csv"

# 2. 正解ラベルマスクの作成 (左側の画像)
print("正解ラベルマスク（プラスチックのみ）を作成しています...")
with open(json_path, 'r') as f:
    json_data = json.load(f)

img_height = json_data['imageHeight']
img_width = json_data['imageWidth']

# labelme形式で数値マスクを生成
labels_in_json = sorted(list(set(shape['label'] for shape in json_data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
numeric_ground_truth_mask, _ = labelme.utils.shapes_to_label((img_height, img_width), json_data['shapes'], label_name_to_value)

# 数値マスクをカラーマスクに変換 (★★★ 背景は描画しない ★★★)
ground_truth_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
for label_name, label_value in label_name_to_value.items():
    # プラスチックラベルのみを色付けする
    if label_name in label_to_color_map:
        color_rgb = (np.array(plt.cm.colors.to_rgb(label_to_color_map[label_name])) * 255).astype(np.uint8)
        ground_truth_mask[numeric_ground_truth_mask == label_value] = color_rgb

# 3. 予測結果マスクの作成 (右側の画像)
print("モデルによる予測とマスク作成を実行しています...")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

df_test = pd.read_csv(test_csv_path)

original_indices = df_test['original_index'].values
X_test = df_test[feature_names_from_model]
y_pred = model.predict(X_test)

predicted_mask_flat = np.zeros((img_height * img_width, 3), dtype=np.uint8)
for label_name, color_name in label_to_color_map.items():
    pred_indices_in_y = np.where(y_pred == label_name)[0]
    if len(pred_indices_in_y) > 0:
        img_indices_to_paint = original_indices[pred_indices_in_y]
        color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
        predicted_mask_flat[img_indices_to_paint] = color_rgb

predicted_mask = predicted_mask_flat.reshape(img_height, img_width, 3)

# 4. 2つの画像を並べてプロット
print("\n--- 結果の比較表示 ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左: 正解ラベル
axes[0].imshow(ground_truth_mask)
axes[0].set_title('Ground Truth (Plastics Only)', fontsize=16)
axes[0].axis('off')

# 右: モデルの予測結果
axes[1].imshow(predicted_mask)
axes[1].set_title('Model Prediction (Plastics Only)', fontsize=16)
axes[1].axis('off')

# 共通の凡例を作成
legend_patches = []
for label_name, color in sorted(label_to_color_map.items()):
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
fig.legend(handles=legend_patches, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %% [markdown]
# ### 混同行列

# %%
import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ユーザー設定 (ここを編集してください) ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")

# --- 検証パターンを選択 ---
# A: 背景ありデータセット / B: プラスチックのみデータセット
# -----------------------------
# ★★★ 今回は「B: プラスチックのみ」の交差検証を可視化 ★★★
# -----------------------------
# 評価に使用するデータセット
evaluation_dataset_folder_name = "MPs_20250905_2" 
# 使用する学習済みモデル
model_path = main_dir / "model_plastics_only_dataset1.joblib"
# 使用するCSVファイル
csv_filename = "pixel_features_plastics_only.csv"
# --------------------------------------------------


# 1. データの準備とモデルの読み込み
print("--- データの準備 ---")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

test_data_path = main_dir / evaluation_dataset_folder_name / "csv" / csv_filename
try:
    df_test = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"エラー: {test_data_path} が見つかりません。パスを確認してください。")
    exit()

# 特徴量 (X_test) と正解ラベル (y_true) を準備
X_test = df_test[feature_names_from_model]
y_true = df_test['label_name']

# 2. モデルによる予測
print("\n--- ラベルを予測 ---")
y_pred = model.predict(X_test)
print("予測が完了しました。")

# 3. 混同行列の計算
# ラベルの順序をアルファベット順で固定
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 4. 混同行列の可視化 (生データ)
print("\n--- 混同行列の可視化 ---")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Plastics Only - Raw Counts)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# 5. 正規化された混同行列の可視化 (行の合計が1になるように正規化)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Normalized Confusion Matrix (Plastics Only - Recall)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()





# %%
import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ユーザー設定 (ここを編集してください) ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")

# --- 検証パターンを選択 ---
# A: 背景ありデータセット / B: プラスチックのみデータセット
# -----------------------------
# ★★★ 今回は「B: プラスチックのみ」の交差検証を可視化 ★★★
# -----------------------------
# 評価に使用するデータセット
evaluation_dataset_folder_name = "MPs_20250911" 
# 使用する学習済みモデル
model_path = main_dir / "model_plastics_only_dataset2.joblib"
# 使用するCSVファイル
csv_filename = "pixel_features_plastics_only.csv"
# --------------------------------------------------


# 1. データの準備とモデルの読み込み
print("--- データの準備 ---")
model = joblib.load(model_path)
feature_names_from_model = model.feature_names_in_

test_data_path = main_dir / evaluation_dataset_folder_name / "csv" / csv_filename
try:
    df_test = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"エラー: {test_data_path} が見つかりません。パスを確認してください。")
    exit()

# 特徴量 (X_test) と正解ラベル (y_true) を準備
X_test = df_test[feature_names_from_model]
y_true = df_test['label_name']

# 2. モデルによる予測
print("\n--- ラベルを予測 ---")
y_pred = model.predict(X_test)
print("予測が完了しました。")

# 3. 混同行列の計算
# ラベルの順序をアルファベット順で固定
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 4. 混同行列の可視化 (生データ)
print("\n--- 混同行列の可視化 ---")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Plastics Only - Raw Counts)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# 5. 正規化された混同行列の可視化 (行の合計が1になるように正規化)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Normalized Confusion Matrix (Plastics Only - Recall)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()



# %% [markdown]
# ---

# %% [markdown]
# # 特徴量重要度から分類に必要な計測時間と精度を調べる

# %%
import pandas as pd
from pathlib import Path

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# プラスチックのみのデータセットを使用
csv_filename = "pixel_features_plastics_only.csv"
importance_filename1 = "importance_plastics_only_from_dataset1.csv"
importance_filename2 = "importance_plastics_only_from_dataset2.csv"
# --------------------

# 1. 2つの特徴量重要度ランキングのファイルを読み込み、統合する
print("--- ステップ1: 共通重要度ランキングの作成 ---")
try:
    imp1_df = pd.read_csv(main_dir / dataset1_folder_name / "csv" / importance_filename1)
    imp2_df = pd.read_csv(main_dir / dataset2_folder_name / "csv" / importance_filename2)

    # 2つのランキングを特徴量名で結合し、平均重要度を計算
    merged_imp = pd.merge(imp1_df, imp2_df, on='feature', suffixes=('_d1', '_d2'))
    merged_imp['average_importance'] = (merged_imp['importance_d1'] + merged_imp['importance_d2']) / 2

    # 平均重要度でソートし、最終的なランキングを作成
    final_importances = merged_imp.sort_values('average_importance', ascending=False)
    sorted_features = final_importances['feature'].tolist()
    
    print("2つのモデルの平均重要度に基づき、特徴量ランキングを再計算しました。")
    print("重要度トップ5の分光画像:")
    print(sorted_features[:5])

except FileNotFoundError as e:
    print(f"エラー: 特徴量重要度ファイルが見つかりません: {e.filename}")
    print("先に「交差検証（プラスチックのみ）」スクリプトを実行して、重要度ファイルを作成してください。")
    exit()

# 2. 検証に使用するデータセットを読み込む
df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
print("\n検証用データセットの読み込みが完了しました。")


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time

# --- ユーザー設定 ---
# 検証する特徴量の枚数のリスト
# num_features_steps = [1, 5, 10, 20, 30, 50, 100, 150, 200]
num_features_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# --------------------

# 結果を保存するためのリスト
results_log = []

def prepare_data(df, features_to_use):
    """指定された特徴量だけを使ってX, yを準備する関数"""
    # label_nameと指定された特徴量だけを抽出
    subset_df = df[['label_name'] + features_to_use]
    X = subset_df.drop(columns=['label_name'])
    y = subset_df['label_name']
    return X, y

# ステップごとにループ
for n_features in num_features_steps:
    start_time = time.time()
    
    # 使用する特徴量を上位から選択
    selected_features = sorted_features[:n_features]
    
    print(f"\n--- 特徴量 {n_features} 枚で交差検証中... ---")
    
    # 1. データセット1で学習 -> データセット2で評価
    X1, y1 = prepare_data(df1, selected_features)
    X2, y2 = prepare_data(df2, selected_features)
    
    model1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model1.fit(X1, y1)
    y_pred1 = model1.predict(X2)
    report1 = classification_report(y2, y_pred1, output_dict=True, zero_division=0)

    # 2. データセット2で学習 -> データセット1で評価
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model2.fit(X2, y2)
    y_pred2 = model2.predict(X1)
    report2 = classification_report(y1, y_pred2, output_dict=True, zero_division=0)
    
    # 3. 両方向のF1スコアの平均を記録
    result_row = {'num_features': n_features}
    all_labels = sorted(list(set(y1) | set(y2))) # 全ラベルのリストを作成
    for label in all_labels:
        f1_1 = report1.get(label, {}).get('f1-score', 0)
        f1_2 = report2.get(label, {}).get('f1-score', 0)
        result_row[label] = (f1_1 + f1_2) / 2.0
    results_log.append(result_row)
    
    end_time = time.time()
    print(f"完了。処理時間: {end_time - start_time:.2f} 秒")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results_log)
results_df.to_csv(main_dir / "crossval_performance_vs_num_features.csv", index=False)
print("\n--- 全ステップの評価が完了しました ---")
print("結果を crossval_performance_vs_num_features.csv に保存しました。")
print(results_df)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# ステップ2で作成したDataFrame (results_df) を使用
df_melted = results_df.melt(id_vars='num_features', var_name='plastic_type', value_name='f1_score')

# グラフの描画
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(18, 10))

sns.lineplot(data=df_melted, x='num_features', y='f1_score', hue='plastic_type', marker='o', palette='tab10')

# グラフの装飾
plt.title('Cross-Validation Performance vs. Number of Top Features', fontsize=20)
plt.xlabel('Number of Spectral Images Used (by importance)', fontsize=14)
plt.ylabel('Average F1-Score (Cross-Validation)', fontsize=14)
plt.xticks(results_df['num_features'].unique(), rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Plastic Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %% [markdown]
# ---

# %% [markdown]
# ### 分光画像の平均画素値から、EEMを作成

# %% [markdown]
# 平均スペクトルデータの作成

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザー設定 ---
folder_name = "MPs_20250911"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1"
# --------------------

# 1. JSONファイルを読み込み、ラベルマスクを作成
print("--- ステップ1: 平均スペクトルデータの作成 ---")
json_path = main_dir / (reference_file_stem + ".json")
with open(json_path, 'r') as f:
    data = json.load(f)

image_size = (data['imageHeight'], data['imageWidth'])
labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}
class_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)

# 2. 各分光画像を読み込み、プラスチックごとに画素値の平均を計算
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
image_files = list(main_dir.glob("*.tiff"))

# 結果を格納する辞書を初期化
mean_spectra = {label: {} for label in labels_in_json}

for image_path in image_files:
    if '-1_Em-1' in image_path.stem: continue
    match = wavelength_pattern.search(image_path.name)
    if not match: continue
    
    ex_wave = int(match.group(1))
    em_wave = int(match.group(2))

    try:
        img = np.asarray(Image.open(image_path))
    except Exception as e:
        print(f"Warning: Failed to process {image_path.name}. Reason: {e}")
        continue

    # 各プラスチック領域の平均値を計算
    for label_name, label_value in label_name_to_value.items():
        mean_intensity = np.mean(img[class_label_mask == label_value])
        mean_spectra[label_name][(ex_wave, em_wave)] = mean_intensity

print("平均スペクトルデータの作成が完了しました。")


# %%
import numpy as np
import matplotlib.pyplot as plt

# --- ユーザー設定 ---
# EEMの軸範囲と刻み幅
EX_MIN, EX_MAX, EX_STEP = 260, 600, 20
EM_MIN, EM_MAX, EM_STEP = 260, 600, 20

# ★★★ プロットする順番をここで指定 ★★★
plastics_to_plot = [
    'PP', 'PC', 'ABS',
    'PS', 'PET', 'HDPE',
    'PVC', 'PMMA', 'LDPE'
]
# --------------------

# EEMの軸を定義
excitation_axis = np.arange(EX_MIN, EX_MAX + EX_STEP, EX_STEP)
emission_axis = np.arange(EM_MIN, EM_MAX + EM_STEP, EM_STEP)

# 各プラスチックについてEEMを作成・描画
num_plastics = len(plastics_to_plot)
ncols = 3
nrows = int(np.ceil(num_plastics / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
if nrows == 1 or ncols == 1:
    axes = np.array(axes).reshape(nrows, ncols)

for i, plastic_name in enumerate(plastics_to_plot):
    # check if the plastic name from the custom list exists in the data
    if plastic_name not in mean_spectra:
        print(f"警告: 指定されたプラスチック '{plastic_name}' のデータが見つかりません。スキップします。")
        # Turn off the axis for the empty plot
        ax = axes[i // ncols, i % ncols]
        ax.axis('off')
        continue

    ax = axes[i // ncols, i % ncols]
    
    # EEM行列の形を(励起, 放射)に入れ替え
    eem_matrix = np.full((len(excitation_axis), len(emission_axis)), np.nan)
    
    # 平均スペクトルデータをEEM行列にマッピング
    for (ex, em), intensity in mean_spectra[plastic_name].items():
        try:
            ex_idx = np.where(excitation_axis == ex)[0][0]
            em_idx = np.where(emission_axis == em)[0][0]
            # 行と列のインデックスを入れ替え
            eem_matrix[ex_idx, em_idx] = intensity
        except IndexError:
            pass
            
    # 散乱光や無効領域をマスク
    for ex_idx, ex in enumerate(excitation_axis):
        for em_idx, em in enumerate(emission_axis):
            if em <= ex:
                 # 行と列のインデックスを入れ替え
                eem_matrix[ex_idx, em_idx] = np.nan

    # ヒートマップを描画
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='black')
    
    # extentの範囲を(放射, 励起)に入れ替え
    im = ax.imshow(eem_matrix, cmap=cmap, origin='lower', 
                   extent=[EM_MIN, EM_MAX, EX_MIN, EX_MAX], aspect='auto')
    
    # 表示範囲の軸を入れ替え
    ax.set_xlim(250, 600)
    ax.set_ylim(250, 600)
    
    ax.set_title(plastic_name)
    # X軸とY軸のラベルを入れ替え
    ax.set_xlabel('Emission (nm)')
    ax.set_ylabel('Excitation (nm)')

    # グリッドを非表示にする
    ax.grid(False)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensity')

# 使わないサブプロットを非表示
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()




