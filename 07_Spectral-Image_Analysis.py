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
# ---

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
# folder_name = "MPs_20250911"
folder_name = "MPs_20250905_2"

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
# ## データセットの生成

# %% [markdown]
# ### background情報あり

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

# 3. ★★★ ラベル情報の処理方法を修正 ★★★
print("\nスペクトルデータの抽出が完了しました。ラベル情報を結合・整形します...")

# 3-1. JSON内のラベルにのみ数値を割り当て（未ラベル領域はデフォルトで0になる）
labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}

# 3-2. ラベルマスクを生成
pixel_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)
pixel_labels_flat = pixel_label_mask.flatten()

# 3-3. 数値とラベル名を対応付け（0は「未ラベル」とする）
value_to_label_name = {v: k for k, v in label_name_to_value.items()}
value_to_label_name[0] = '_unlabeled_' # ラベル付けされていない領域を明示的に命名

pixel_features_df['label_name'] = pd.Series(pixel_labels_flat).map(value_to_label_name)

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
# # t-SNE データセットの可視化

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
# # 2つのデータセットを用いた交差検証

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
from sklearn.metrics import classification_report
import joblib
# import time

def prepare_data(df):
    """背景データを含め、特徴量とラベルに分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    X = df.drop(columns=['label_value', 'label_name'])
    y = df['label_name']
    return X, y

def train_and_save(X_train, y_train, model_save_path):
    print("\n--- ランダムフォレストモデルの学習を開始 ---")
    
    # モデルの学習
    # 修正点: class_weight='balanced' を追加
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # モデルの保存
    joblib.dump(model, model_save_path)

    print(f"学習済みモデルを保存しました: {model_save_path}")
    
    return model

# データセットの読み込み (前のセクションで定義されたパスを使用)
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"
dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_with_background.csv"

df1 = pd.read_csv(dataset1_csv_path)
df2 = pd.read_csv(dataset2_csv_path)

X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# モデル1を学習・保存
train_and_save(X1, y1, main_dir / "model_trained_on_dataset1.joblib")
print(f"データセット1（{dataset1_folder_name}）学習完了")


# モデル2を学習・保存
train_and_save(X2, y2, main_dir / "model_trained_on_dataset2.joblib")
print(f"データセット2（{dataset2_folder_name}）学習完了")

# %%
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import time

def prepare_data(df):
    """背景データを含め、特徴量とラベルに分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    X = df.drop(columns=['label_value', 'label_name'])
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
    
    # 特徴量重要度を計算
    feature_importances = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    return importance_df, model # 修正点: importance_df と model を返すように変更

# データセットの読み込み
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"
dataset2_csv_path = main_dir / dataset2_folder_name / "csv" / "pixel_features_with_background.csv"

df1 = pd.read_csv(dataset1_csv_path)
df2 = pd.read_csv(dataset2_csv_path)

X1, y1 = prepare_data(df1)
X2, y2 = prepare_data(df2)

# 交差検証1: モデル1を評価
# evaluate_model関数の返り値を model1 に格納
importance1, model1 = evaluate_model(main_dir / "model_trained_on_dataset1.joblib", X2, y2, "モデル1 (Dataset1で学習)")

# 交差検証2: モデル2を評価
# evaluate_model関数の返り値を model2 に格納
importance2, model2 = evaluate_model(main_dir / "model_trained_on_dataset2.joblib", X1, y1, "モデル2 (Dataset2で学習)")


# --- 結果を保存するコードブロック ---
# この部分で model1 と model2 変数が使用可能になっている
data1_output_dir = main_dir / dataset1_folder_name
data2_output_dir = main_dir / dataset2_folder_name

# output_dir.mkdir(parents=True, exist_ok=True)

# 精度レポートをテキストファイルとして保存
# 交差検証1の結果
with open(data1_output_dir / "classification_report_model1.txt", "w") as f:
    f.write(classification_report(y2, model1.predict(X2)))

# 交差検証2の結果
with open(data2_output_dir / "classification_report_model2.txt", "w") as f:
    f.write(classification_report(y1, model2.predict(X1)))

# 特徴量重要度をCSVファイルとして保存
importance1.to_csv(data1_output_dir / "csv" / "importance_from_dataset.csv", index=False)
importance2.to_csv(data2_output_dir / "csv" / "importance_from_dataset.csv", index=False)

print("\n--- 全ての処理が完了しました ---")
print(f"精度レポートと重要度ランキングは保存されました")

# %%
# # --- 結果を保存するコードブロック ---
# output_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/results")
# output_dir.mkdir(parents=True, exist_ok=True)

# # 精度レポートをテキストファイルとして保存
# # 交差検証1の結果
# with open(output_dir / "classification_report_model1.txt", "w") as f:
#     f.write(classification_report(y2, model1.predict(X2)))

# # 交差検証2の結果
# with open(output_dir / "classification_report_model2.txt", "w") as f:
#     f.write(classification_report(y1, model2.predict(X1)))

# # 特徴量重要度をCSVファイルとして保存
# importance1.to_csv(output_dir / "importance_from_dataset1.csv", index=False)
# importance2.to_csv(output_dir / "importance_from_dataset2.csv", index=False)

# print("\n--- 全ての処理が完了しました ---")
# print(f"精度レポートと重要度ランキングは以下のディレクトリに保存されました: {output_dir}")

# %% [markdown]
# ## 分類結果の可視化

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import labelme

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセットのフォルダ名（テストデータとしてMPs_20250905_2を使用）
test_dataset_folder_name = "MPs_20250905_2" 
# 学習済みモデルのパス（Dataset1で学習し、resultsフォルダに保存したもの）
model_path = main_dir / "model_trained_on_dataset1.joblib"
# ------------------------------

# 1. データの準備とモデルの読み込み
print("--- データの準備 ---")

# ラベル名と色を対応付けるための辞書を定義
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime", "_background_": "gray"
}

# モデルを読み込み、学習に使用された全波長を取得
model = joblib.load(model_path)
feature_names = model.feature_names_in_

# 評価対象の全波長データセットを読み込む
test_data_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_with_background.csv"
try:
    df_test = pd.read_csv(test_data_path, index_col=0) # 修正点：index_col=0 を追加
except FileNotFoundError:
    print(f"エラー: {test_data_path} が見つかりません。パスを確認してください。")
    exit()

# 特徴量（X_test）を準備
X_test = df_test[feature_names]

# 元画像のパスを取得（可視化用）
reference_image_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.tiff"
original_image = np.asarray(Image.open(reference_image_path))
json_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.json"

# 2. モデルによるピクセルごとの分類
print("\n--- ピクセルごとのラベルを予測 ---")
y_pred = model.predict(X_test)
print("予測が完了しました。")

# 3. JSONファイルから完全なマスクを生成し、可視化
print("\n--- 予測結果の可視化 ---")

# 元画像の形状（高さと幅）を取得
img_height, img_width = original_image.shape

# 予測結果を画像全体にマッピング
y_pred_full = np.array([''] * (img_height * img_width), dtype=object) # 空の文字列で初期化
y_pred_full[df_test.index] = y_pred

# JSONからotherラベルを特定
with open(json_path, 'r') as f:
    json_data = json.load(f)

label_name_to_value = {"_background_": 0}
for shape in json_data["shapes"]:
    label_name = shape["label"]
    if label_name not in label_name_to_value:
        label_name_to_value[label_name] = len(label_name_to_value)

full_mask, _ = labelme.utils.shapes_to_label(
    original_image.shape, json_data["shapes"], label_name_to_value
)

# カラーマスクの生成
predicted_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

for label_name, color_name in label_to_color_map.items():
    indices = np.where(y_pred_full == label_name)[0]
    color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
    predicted_mask.reshape(-1, 3)[indices] = color_rgb

# otherラベルの領域を黒く塗りつぶす
other_indices = np.where(full_mask == label_name_to_value.get('other', -1))[0]
predicted_mask.reshape(-1, 3)[other_indices] = [0, 0, 0]

# 図の作成と表示
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(predicted_mask)
axes[1].set_title('Predicted Labels (from Dataset1 model)')
axes[1].axis('off')

# 凡例の作成
legend_patches = []
for label_name, color in label_to_color_map.items():
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("\n可視化が完了しました。")

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
# 評価に使用するデータセットのフォルダ名（テストデータとしてMPs_20250911を使用）
test_dataset_folder_name = "MPs_20250911" 
# 学習済みモデルのパス（Dataset2で学習し、resultsフォルダに保存したもの）
model_path = main_dir / "model_trained_on_dataset2.joblib"
# ------------------------------

# 1. データの準備とモデルの読み込み
print("--- データの準備 ---")

# ラベル名と色を対応付けるための辞書を定義
label_to_color_map = {
    "ABS": "red", "HDPE": "blue", "LDPE": "green", "PC": "yellow", "PET": "purple",
    "PMMA": "orange", "PP": "cyan", "PS": "magenta", "PVC": "lime", "_background_": "gray"
}
labels = list(label_to_color_map.keys())

# モデルを読み込み、学習に使用された全波長を取得
model = joblib.load(model_path)
feature_names = model.feature_names_in_

# 評価対象の全波長データセットを読み込む
test_data_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_with_background.csv"
try:
    df_test = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"エラー: {test_data_path} が見つかりません。パスを確認してください。")
    exit()

# 特徴量（X_test）を準備
X_test = df_test[feature_names]

# 元画像のパスを取得（可視化用）
reference_image_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.tiff"
original_image = np.asarray(Image.open(reference_image_path))

# 2. モデルによるピクセルごとの分類
print("\n--- ピクセルごとのラベルを予測 ---")
y_pred = model.predict(X_test)
print("予測が完了しました。")

# 3. 予測結果をカラーマスクとして可視化
print("\n--- 予測結果の可視化 ---")

# 元画像の形状（高さと幅）を取得
img_height, img_width = original_image.shape

# 予測結果からカラーマスクを生成
predicted_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

# ラベル名と色をマッピングし、マスクに適用
for label_name, color_name in label_to_color_map.items():
    indices = np.where(y_pred == label_name)[0]
    color_rgb = (np.array(plt.cm.colors.to_rgb(color_name)) * 255).astype(np.uint8)
    
    # 予測マスクの作成
    predicted_mask.reshape(-1, 3)[indices] = color_rgb

# 図の作成と表示
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# 1. 元画像の表示
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# 2. 予測結果の可視化
axes[1].imshow(predicted_mask)
axes[1].set_title('Predicted Labels (from Dataset2 model)')
axes[1].axis('off')

# 凡例の作成
legend_patches = []
for label_name, color in label_to_color_map.items():
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("\n可視化が完了しました。")

# %% [markdown]
# ---

# %% [markdown]
# # 上位10個の分光画像のみを使用して交差検証
# 

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

dataset1_dir = main_dir / dataset1_folder_name
dataset2_dir = main_dir / dataset2_folder_name

n_features = 10  # 上位何個の波長を使うか指定
#-------------------------------------------------------------------

# print(Path(data1_output_dir / "importance_from_dataset.csv"))

def create_filtered_dataset(dataset_dir, n_features):
    """
    指定されたフォルダの重要度ランキングから上位N波長を抽出し、
    新しいデータセットを生成する関数
    """
    importance_csv_path = dataset_dir / "importance_from_dataset.csv"
    pixel_features_all_data_path = dataset_dir / "csv" / "pixel_features_with_background.csv"

    # importance_csv_path = dataset_dir / "csv" / "importance_from_dataset.csv"
    # pixel_features_all_data_path = dataset_dir / "csv" / "pixel_features_with_background.csv"

    # 特徴量重要度ランキングを読み込み
    importance_df = pd.read_csv(importance_csv_path)
    # 上位 n_features の波長名を取得
    top_features = importance_df['feature'].head(n_features).tolist()

    # ピクセル単位データを読み込み
    df_pixels = pd.read_csv(pixel_features_all_data_path)

    # 必要な列だけ抽出
    df_selected = df_pixels[['label_value', 'label_name'] + top_features]
    
    print(f"上位{n_features}波長で新しいデータセットを作成しました。: {importance_csv_path}")
    print(f"選択された波長: {top_features}")
    
    return df_selected

# データセット1と2を、それぞれの上位10波長でフィルタリング
df_selected1 = create_filtered_dataset(dataset1_dir, n_features)
df_selected2 = create_filtered_dataset(dataset2_dir, n_features)

print("\n--- 全てのデータセットのフィルタリングが完了しました ---\n")

# %%
def prepare_data(df):
    """背景データを含め、特徴量とラベルに分割するヘルパー関数"""
    if df.empty:
        raise ValueError("データセットにピクセルデータがありません。")
    X = df.drop(columns=['label_value', 'label_name'])
    y = df['label_name']
    return X, y

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    print(f"--- {model_name}の学習と評価 ---")
    
    # 訓練データとテストデータで波長の列を揃える
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]

    # ランダムフォレストモデルの学習
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 評価
    y_pred = model.predict(X_test)
    print('--- 精度評価レポート ---')
    print(classification_report(y_test, y_pred))

    return model

# データセット1と2を学習・評価用に分割
try:
    X1, y1 = prepare_data(df_selected1)
    X2, y2 = prepare_data(df_selected2)
except ValueError as e:
    print(f"エラー: {e}")
    exit()

# 交差検証1: Dataset1で学習し、Dataset2でテスト
model1 = train_and_evaluate(X1, y1, X2, y2, "モデル1 (Dataset1で学習)")

# 交差検証2: Dataset2で学習し、Dataset1でテスト
model2 = train_and_evaluate(X2, y2, X1, y1, "モデル2 (Dataset2で学習)")

# %%
# モデルの保存
output_dir1 = dataset1_dir / "results_top10_features"
output_dir1.mkdir(parents=True, exist_ok=True)

joblib.dump(model1, output_dir1 / "model_trained_on_dataset1.joblib")
print(f"学習済みモデルは以下のディレクトリに保存されました: {output_dir1}")


output_dir2 = dataset1_dir / "results_top10_features"
output_dir2.mkdir(parents=True, exist_ok=True)

joblib.dump(model2, output_dir2 / "model_trained_on_dataset2.joblib")
print(f"学習済みモデルは以下のディレクトリに保存されました: {output_dir2}")


print("\n--- 全ての処理が完了しました ---")

# %% [markdown]
# ---
# ## 可視化

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
test_dataset_folder_name = "MPs_20250911" 

# 評価に使用するモデルのパス（例：Dataset2で学習したモデル）
model_path = main_dir / test_dataset_folder_name / "results_top10_features" / "model_trained_on_dataset2.joblib"
# ------------------------------

# 1. データの準備とモデルの読み込み
print("--- データの準備 ---")

# ラベル名と色を対応付けるための辞書を定義
label_to_color_map = {
    "ABS": "red",
    "HDPE": "blue",
    "LDPE": "green",
    "PC": "yellow",
    "PET": "purple",
    "PMMA": "orange",
    "PP": "cyan",
    "PS": "magenta",
    "PVC": "lime",
    "_background_": "gray"
}
labels = list(label_to_color_map.keys())

# 特徴量抽出に必要な波長リストを取得
model = joblib.load(model_path)
feature_names = model.feature_names_in_

# 評価対象のデータセットを読み込む
test_data_path = main_dir / test_dataset_folder_name / "csv" / "pixel_features_top10_wavelengths_with_background.csv"
try:
    df_test = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"エラー: {test_data_path} が見つかりません。パスを確認してください。")
    exit()

# 2. モデルによる予測
print("\n--- ピクセルごとのラベルを予測 ---")
# 特徴量（X）と正解ラベル（y）に分割
X_test = df_test[feature_names]
y_true = df_test['label_name']
y_pred = model.predict(X_test)
print("予測が完了しました。")

# 3. 予測結果の可視化
print("\n--- 予測結果の可視化 ---")

# 元画像のパスを取得（可視化用）
reference_image_path = main_dir / test_dataset_folder_name / f"{test_dataset_folder_name}_Ex-1_Em-1_ET300_step1.tiff"
original_image = np.asarray(Image.open(reference_image_path))

# 元画像の形状（高さと幅）を取得
img_height, img_width = original_image.shape

# 予測結果からカラーマスクを生成
pred_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
true_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

# ラベル名と色をマッピング
for i, label_name in enumerate(labels):
    color = plt.cm.get_cmap('jet', len(labels))(i)
    color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
    
    # 予測マスクの作成
    pred_indices = np.where(y_pred == label_name)[0]
    pred_mask.reshape(-1, 3)[pred_indices] = color_rgb

    # 正解マスクの作成
    true_indices = np.where(y_true == label_name)[0]
    true_mask.reshape(-1, 3)[true_indices] = color_rgb

# 図の作成
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

# 1. 元画像の表示
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# 2. 予測結果の可視化
axes[1].imshow(pred_mask)
axes[1].set_title('Predicted Labels')
axes[1].axis('off')

# 3. 正解ラベルの可視化
axes[2].imshow(true_mask)
axes[2].set_title('True Labels')
axes[2].axis('off')

# 凡例の作成
legend_patches = []
for label_name, color in label_to_color_map.items():
    legend_patches.append(mpatches.Patch(color=color, label=label_name))
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# ----

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# # --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
# # ------------------------------

# 作成されたピクセル単位のデータセットを読み込む
dataset1_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"
dataset2_csv_path = main_dir / dataset1_folder_name / "csv" / "pixel_features_with_background.csv"

try:
    df_pixels = pd.read_csv(output_csv_path)
except FileNotFoundError:
    print(f"エラー: {output_csv_path} が見つかりません。")
    exit()

print("データセットを正常に読み込みました。")


# 背景（_background_）データを除外して学習データを作成
df_labeled_only = df_pixels[df_pixels['label_name'] != '_background_']

if df_labeled_only.empty:
    print("警告: データセットにラベル付けされたプラスチックのピクセルがありません。")
    exit()

# 特徴量（X）と正解ラベル（y）に分割
X = df_labeled_only.drop(columns=['label_value', 'label_name'])
y = df_labeled_only['label_name']

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n学習データセットのサイズ: {len(X_train)} 件")
print(f"テストデータセットのサイズ: {len(X_test)} 件")

# ランダムフォレストモデルの学習
print("\nランダムフォレストモデルの学習を開始します...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("学習が完了しました。")

# 学習済みモデルをjoblib形式で保存
model_save_path = main_dir / "random_forest_model_no_scatter.joblib"
joblib.dump(model, model_save_path)
print(f"\n学習済みモデルを保存しました: {model_save_path}")

# テストデータで予測を行い、精度を評価
y_pred = model.predict(X_test)
print('\n--- 精度評価レポート ---')
print(classification_report(y_test, y_pred))

# 特徴量の重要度を計算し、ランキング形式で表示
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# 保存先を指定
importance_csv_path = main_dir / "csv" /"selected_wavelengths_importance_with_background.csv"
# 親ディレクトリが存在しない場合は作成
importance_csv_path.parent.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(importance_csv_path, index=False)

print(f"\n特徴量重要度ランキングを保存しました: {importance_csv_path}")
print("\n上位5件の特徴量:")
print(importance_df.head())


# %% [markdown]
# ---
# # 選択した波長の分光画像からの画像分類

# %% [markdown]
# ## データセットの作成

# %% [markdown]
# ### 背景あり

# %%
import pandas as pd
from pathlib import Path

# --- ユーザーが設定する項目 ---
folder_name = "MPs_20250911"
# folder_name = "MPs_20250905_2"

# プロジェクトのメインディレクトリとファイル名を指定
file_stem =f"{folder_name}_Ex-1_Em-1_ET300_step1"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）

n_features = 10  # 上位何個の波長を使うか指定
importance_csv_path = main_dir / "csv" / "selected_wavelengths_importance_with_background.csv"  # 波長選択の結果
pixel_features_all_data_path = main_dir / "csv" / "pixel_features_with_background.csv"  #  ラベル域の全データセット
#-------------------------------------------------------------------


# 1. 特徴量重要度ランキングを読み込み
importance_df = pd.read_csv(importance_csv_path)

# 上位 n_features の波長名を取得
top_features = importance_df['feature'].head(n_features).tolist()
print(f"選択された上位 {n_features} 波長: {top_features}")

# 2. ピクセル単位データを読み込み
df_pixels = pd.read_csv(pixel_features_all_data_path)

# 背景を除外（ラベル名が _background_ の行を削除）
df_labeled_only = df_pixels[df_pixels['label_name'] != '_background_']

# 3. 必要な列だけ抽出
df_selected = df_labeled_only[['label_value', 'label_name'] + top_features]

# 4. 保存
output_csv_path = main_dir / "csv" / f"pixel_features_top{n_features}_wavelengths_with_background.csv"
df_selected.to_csv(output_csv_path, index=False)
print(f"データセットを保存しました: {output_csv_path}")
print(df_selected.head())

# ラベルごとのピクセル数を確認
print("\nラベルごとのピクセル数:")
print(df_selected['label_name'].value_counts())


# --- プレビュー表示 ---
print("\n=== データセット情報 ===")
print(f"全体のサイズ: {df_selected.shape[0]} サンプル, {df_selected.shape[1]} 列")

print("\n=== 先頭5行 ===")
print(df_selected.head())

print("\n=== ラベルごとのピクセル数 ===")
print(df_selected['label_name'].value_counts())


# %% [markdown]
# ### 背景なし

# %%
import pandas as pd
from pathlib import Path

# --- ユーザーが設定する項目 ---
# folder_name = "MPs_20250911"
folder_name = "MPs_20250905_2"

# プロジェクトのメインディレクトリとファイル名を指定
file_stem =f"{folder_name}_Ex-1_Em-1_ET300_step1"
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）

n_features = 10 # 上位何個の波長を使うか指定
importance_csv_path = main_dir / "csv" / "selected_wavelengths_importance_no_background.csv" # 波長選択の結果
pixel_features_all_data_path = main_dir / "csv" / "pixel_features_no_background.csv" #  ラベル域の全データセット
#-------------------------------------------------------------------

# 1. 特徴量重要度ランキングを読み込み
try:
    importance_df = pd.read_csv(importance_csv_path)
except FileNotFoundError:
    print(f"Error: {importance_csv_path} が見つかりません。")
    exit()

# 上位 n_features の波長名を取得
top_features = importance_df['feature'].head(n_features).tolist()
print(f"選択された上位 {n_features} 波長: {top_features}")

# 2. ピクセル単位データを読み込み
try:
    df_pixels = pd.read_csv(pixel_features_all_data_path)
except FileNotFoundError:
    print(f"Error: {pixel_features_all_data_path} が見つかりません。")
    exit()

# 背景データを含めるため、背景除外処理をスキップ
df_labeled_only = df_pixels.copy()

# 3. 必要な列だけ抽出
df_selected = df_labeled_only[['label_value', 'label_name'] + top_features]

# 4. 保存
output_csv_path = main_dir / "csv" / f"pixel_features_top{n_features}_wavelengths_no_background.csv"
output_csv_path.parent.mkdir(parents=True, exist_ok=True)
df_selected.to_csv(output_csv_path, index=False)
print(f"データセットを保存しました: {output_csv_path}")


# --- プレビュー表示 ---
print("\n=== データセット情報 ===")
print(f"全体のサイズ: {df_selected.shape[0]} サンプル, {df_selected.shape[1]} 列")

print("\n=== 先頭5行 ===") 
print(df_selected.head())

print("\n=== ラベルごとのピクセル数 ===")
print(df_selected['label_name'].value_counts())

# %% [markdown]
# ## RandomForestで学習・評価

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 説明変数と目的変数
X = df_selected.drop(columns=['label_value', 'label_name'])
y = df_selected['label_name']

# 学習データとテストデータ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# RandomForest（クラス不均衡対応）
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # 少数クラスを重視
)
model.fit(X_train, y_train)

# 精度評価
y_pred = model.predict(X_test)
print('--- 精度評価レポート（重要波長のみ・クラス不均衡対応） ---')
print(classification_report(y_test, y_pred))


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- ユーザー設定 ---
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
# reference_file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1"   # 元画像（JSON・tiff）の基準
visualize_file_stem = f"{folder_name}_Ex260_Em280_ET10000_step1"  # 可視化対象画像
pixel_features_all_data_path = main_dir / "csv" / "pixel_features_no_background.csv"

# top_features = [
#     'Ex360_Em380', 'Ex320_Em360', 'Ex380_Em420', 'Ex260_Em360', 'Ex300_Em380',
#     'Ex360_Em460', 'Ex300_Em360', 'Ex280_Em380', 'Ex340_Em380', 'Ex340_Em400'
# ]
# --------------------

# 1. データ読み込み
df_pixels = pd.read_csv(pixel_features_all_data_path)

# 2. 背景も含むデータでモデル学習
df_labeled_only = df_pixels[df_pixels['label_name'] != '_background_']  # 学習はラベルのみ

X = df_labeled_only[top_features]
y = df_labeled_only['label_name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
print('--- 精度評価レポート（重要波長のみ・クラス不均衡対応） ---')
print(classification_report(y_test, y_pred_test))

# 3. JSONからラベル情報を取得
json_path = main_dir / (reference_file_stem + ".json")
with open(json_path, 'r') as f:
    data = json.load(f)

labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i+1 for i, label in enumerate(labels_in_json)}
label_name_to_value['_background_'] = 0
value_to_label_name = {v: k for k, v in label_name_to_value.items()}

# 4. 可視化対象画像のサイズ
image_path = main_dir / (reference_file_stem + ".tiff")
original_img = np.asarray(Image.open(image_path))
height, width = original_img.shape

# 5. 全ピクセル（背景も含む）に対して予測
X_all = df_pixels[top_features]  # 学習時と同じ列順・列名
y_pred_all = model.predict(X_all)

# 数値ラベルに変換（背景は0のまま）
predicted_mask_flat = pd.Series(y_pred_all).map(lambda x: label_name_to_value.get(x, 0)).values
predicted_mask = predicted_mask_flat.reshape(height, width)

# 背景を透明にするマスク
mask_alpha = np.where(predicted_mask == 0, 0.0, 0.6)

# 6. カラーマップ設定
colors = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
          '#00FFFF', '#FFA500', '#800080', '#A52A2A']  # 適宜追加
cmap = ListedColormap(colors[:len(value_to_label_name)])

legend_patches = []
for label_value, label_name in value_to_label_name.items():
    if label_name == '_background_':
        continue
    color_rgb = np.array([int(colors[label_value][i:i+2],16)/255. for i in (1,3,5)])
    legend_patches.append(mpatches.Patch(color=color_rgb, label=label_name))

# 7. 可視化
fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(np.asarray(Image.open(main_dir / (visualize_file_stem + ".tiff"))), cmap='gray')
ax.imshow(predicted_mask, cmap=cmap, alpha=mask_alpha, vmin=0, vmax=len(value_to_label_name)-1)
ax.set_title(f'Predicted Classification (Selected Wavelengths)')
ax.axis('off')
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# # --- ユーザー設定 ---
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
# reference_file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1"   # 元画像（JSON・tiff）の基準
# visualize_file_stem = "MPs_20250905_2_Ex260_Em280_ET10000_step1"  # 可視化対象画像
# top_features = [
#     'Ex360_Em380', 'Ex320_Em360', 'Ex380_Em420', 'Ex260_Em360', 'Ex300_Em380',
#     'Ex360_Em460', 'Ex300_Em360', 'Ex280_Em380', 'Ex340_Em380', 'Ex340_Em400'
# ]
# --------------------

# 1. データ読み込み
pixel_features_all_data_path = main_dir / "csv" / "pixel_features_no_background.csv"
df_pixels = pd.read_csv(pixel_features_all_data_path)

# 2. 学習用データ（背景は除外）
df_labeled_only = df_pixels[df_pixels['label_name'] != '_background_']
X = df_labeled_only[top_features]
y = df_labeled_only['label_name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 精度評価
y_pred_test = model.predict(X_test)
print('--- 精度評価レポート（重要波長のみ・クラス不均衡対応） ---')
print(classification_report(y_test, y_pred_test))

# 3. JSONからラベル情報取得
json_path = main_dir / (reference_file_stem + ".json")
with open(json_path, 'r') as f:
    data = json.load(f)

labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i+1 for i, label in enumerate(labels_in_json)}  # プラスチックのみ
value_to_label_name = {v: k for k, v in label_name_to_value.items()}

# 4. 元画像サイズ取得
image_path = main_dir / (visualize_file_stem + ".tiff")
original_img = np.asarray(Image.open(image_path))
height, width = original_img.shape

# 5. 背景は除外して予測
df_for_pred = df_pixels[df_pixels['label_name'] != '_background_']
X_all = df_for_pred[top_features]
y_pred_all = model.predict(X_all)

# 6. 予測結果を元画像サイズに埋め込む
predicted_mask = np.zeros((height, width), dtype=int)
predicted_flat = pd.Series(y_pred_all).map(lambda x: label_name_to_value.get(x, 0)).values
predicted_mask_flat_index = np.flatnonzero(df_pixels['label_name'] != '_background_')
predicted_mask.ravel()[predicted_mask_flat_index] = predicted_flat

# 7. カラーマップ設定
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
          '#00FFFF', '#FFA500', '#800080', '#A52A2A', '#FFC0CB']  # 適宜追加
cmap = ListedColormap(colors[:len(value_to_label_name)])

legend_patches = []
for label_value, label_name in value_to_label_name.items():
    color_rgb = np.array([int(colors[label_value-1][i:i+2],16)/255. for i in (1,3,5)])
    legend_patches.append(mpatches.Patch(color=color_rgb, label=label_name))

# 8. 可視化
fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(original_img, cmap='gray')
# 背景以外のみマスクを重ねる
mask_alpha = np.where(predicted_mask==0, 0, 0.6)
ax.imshow(predicted_mask, cmap=cmap, alpha=mask_alpha, vmin=0, vmax=len(value_to_label_name))
ax.set_title(f'Predicted Classification (Selected Wavelengths)')
ax.axis('off')
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


# %% [markdown]
# ---


