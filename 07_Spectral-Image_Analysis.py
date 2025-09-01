# %%
import numpy as np
import labelme
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path

# プロジェクトのメインディレクトリ
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")

# 画像とJSONファイルのパス
json_path = main_dir / "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.json"
image_path = main_dir / "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.tiff"
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
import labelme
import json
from PIL import Image
from pathlib import Path

# プロジェクトのメインディレクトリとファイル名を指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
file_stem = "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1"

json_path = main_dir / (file_stem + ".json")
image_path = main_dir / (file_stem + ".tiff")

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
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 画像ファイルのパスを指定してください
# 基準画像のパス
reference_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.tiff")
# 比較したい分光画像のパス（例）
spectral_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex260_Em280_ET20000_step1.tiff")

# 画像を読み込み
try:
    ref_img = np.asarray(Image.open(reference_image_path))
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

# 画像ファイルのパスを指定
reference_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.tiff")
spectral_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826/MPs_15cm_20250826_Ex260_Em280_ET20000_step1.tiff")

# 画像を読み込み
try:
    ref_img = np.asarray(Image.open(reference_image_path))
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
# ## ラベル単位での分類(修正版)

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
reference_file_stem = "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1"
# ------------------------------

# 1. 基準JSONファイルを読み込む
json_path = main_dir / (reference_file_stem + ".json")
print(f"Loading reference JSON file from: {json_path}")
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: JSONファイルが見つかりません。パスを確認してください。")
    exit()

labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0

all_results = []
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')

# 2. フォルダ内の分光画像ファイルだけを対象にループ処理
image_files = list(main_dir.glob("*Ex*_Em*.tiff"))
image_files = [f for f in image_files if '-1-Em-1-' not in f.stem]

if not image_files:
    print("Error: 指定されたパターンに一致する分光画像ファイルが見つかりません。")
    exit()

print(f"\nFound {len(image_files)} spectral image files to process.")

# 各プラスチック片にユニークなIDを付与
shapes_with_ids = [{**shape, 'id': f"{shape['label']}_{idx}"} for idx, shape in enumerate(data['shapes'])]

for image_path in image_files:
    try:
        match = wavelength_pattern.search(image_path.name)
        if not match:
            print(f"Warning: Wavelength pattern not found in {image_path.name}. Skipping.")
            continue
        ex_wavelength = int(match.group(1))
        em_wavelength = int(match.group(2))

        img = np.asarray(Image.open(image_path))
        lbl, _ = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        
        # 各プラスチック片（shape）ごとに処理
        for shape_obj in shapes_with_ids:
            label_name = shape_obj['label']
            shape_id = shape_obj['id']
            label_value = label_name_to_value[label_name]
            
            # マスクを作成
            shape_mask = (lbl == label_value)

            pixel_values = img[shape_mask]

            if len(pixel_values) > 0:
                result = {
                    'image_name': image_path.name,
                    'label': label_name,
                    'shape_id': shape_id,
                    'Ex_wavelength': ex_wavelength,
                    'Em_wavelength': em_wavelength,
                    'pixel_count': len(pixel_values),
                    'mean': np.mean(pixel_values),
                    'std_dev': np.std(pixel_values),
                    'max_value': np.max(pixel_values),
                    'min_value': np.min(pixel_values)
                }
                all_results.append(result)
    except Exception as e:
        print(f"Warning: Failed to process {image_path.name}. Reason: {e}")
        continue

# 3. すべての結果をデータフレームに整理し、CSVとして保存
if all_results:
    df = pd.DataFrame(all_results)
    output_csv_path = main_dir / "combined_spectral_features_with_ids.csv"
    df.to_csv(output_csv_path, index=False)
    print(f'\n全分光データの統合データセットが作成されました: {output_csv_path}')
    print('\nデータセットのプレビュー:')
    print(df.head())
else:
    print("Warning: No data was successfully processed. Check your files and paths.")

# %%
import pandas as pd
from pathlib import Path

# 作成された統合データセットのパスを指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
output_csv_path = main_dir / "combined_spectral_features_with_ids.csv"

# データセットを読み込み
try:
    df = pd.read_csv(output_csv_path)
except FileNotFoundError:
    print(f"Error: {output_csv_path} が見つかりません。")
    exit()

# 'mean'値の列名を 'Ex' と 'Em' の波長情報を使って動的に作成
df['feature_name'] = 'mean_' + 'Ex' + df['Ex_wavelength'].astype(str) + '_Em' + df['Em_wavelength'].astype(str)

# 不要な列を削除
df_pivot = df.drop(columns=['Ex_wavelength', 'Em_wavelength', 'image_name', 'pixel_count', 'std_dev', 'max_value', 'min_value'])

# pivot_tableを使ってデータを「横長」に変換
# indexに 'shape_id' と 'label' を指定して、各プラスチック片をユニークな行として扱う
df_profile = df_pivot.pivot_table(
    index=['shape_id', 'label'],
    columns='feature_name',
    values='mean',
    aggfunc='mean'
).reset_index()

# 欠損値があれば0で埋める
df_profile = df_profile.fillna(0)

# 新しいデータセットをCSVファイルとして保存
profile_csv_path = main_dir / "spectral_profiles_per_piece.csv"
df_profile.to_csv(profile_csv_path, index=False)

print(f'プラスチック片ごとの分光プロファイルデータセットが作成されました: {profile_csv_path}')
print('\nデータセットのプレビュー:')
print(df_profile.head())

# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 作成された分光プロファイルデータセットのパスを指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
profile_csv_path = main_dir / "spectral_profiles_per_piece.csv"

# データセットを読み込み
try:
    df_profile = pd.read_csv(profile_csv_path)
except FileNotFoundError:
    print(f"Error: {profile_csv_path} が見つかりません。")
    exit()

# ラベルとshape_idのユニークな組み合わせを取得し、これを分割
unique_samples = df_profile[['shape_id', 'label']].drop_duplicates()

# 訓練用とテスト用に分割
# ここでは、プラスチックの種類ごとにサンプルを均等に分割
X_train_samples, X_test_samples, y_train_samples, y_test_samples = train_test_split(
    unique_samples['shape_id'],
    unique_samples['label'],
    test_size=0.3,
    random_state=42,
    stratify=unique_samples['label']
)

# 分割したサンプルIDを使って、元のデータフレームから訓練・テストデータを抽出
X_train = df_profile[df_profile['shape_id'].isin(X_train_samples)].drop(columns=['shape_id', 'label'])
y_train = df_profile[df_profile['shape_id'].isin(X_train_samples)]['label']
X_test = df_profile[df_profile['shape_id'].isin(X_test_samples)].drop(columns=['shape_id', 'label'])
y_test = df_profile[df_profile['shape_id'].isin(X_test_samples)]['label']


# RandomForestClassifierモデルを初期化し、学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# テストデータに対する予測を実行
y_pred = model.predict(X_test)

# 精度評価レポートを表示
print('--- 精度評価レポート（正しい分割による評価） ---')
print(classification_report(y_test, y_pred))

# 特徴量の重要度を取得
feature_importances = model.feature_importances_
feature_names = X_train.columns

# 重要度をデータフレームに整理し、重要度が高い順に並べる
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# 重要度を可視化
plt.figure(figsize=(15, 8))
plt.bar(importance_df['feature'], importance_df['importance'])
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.title('Feature Importances based on Spectral Profiles')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

print("\n特徴量の重要度ランキング（トップ10）:")
print(importance_df.head(10))

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
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
reference_file_stem = "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）
# ------------------------------

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

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
reference_file_stem = "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）
# ------------------------------

# 1. 基準JSONファイルを読み込む
json_path = main_dir / (reference_file_stem + ".json")
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: JSONファイルが見つかりません。パスを確認してください。")
    exit()

labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0

# 2. すべての分光画像の画素値をピクセル単位で抽出
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
image_files = list(main_dir.glob("*Ex*_Em*.tiff"))
image_files = [f for f in image_files if '-1-Em-1-' not in f.stem]

if not image_files:
    print("Error: 指定されたパターンに一致する分光画像ファイルが見つかりません。")
    exit()

pixel_features_df = pd.DataFrame()
processed_files_count = 0
image_size = None # 画像サイズを動的に取得するための変数

for image_path in image_files:
    try:
        img = np.asarray(Image.open(image_path))
        # 初回ループ時に画像サイズを取得
        if image_size is None:
            image_size = img.shape
            print(f"画像サイズを検出しました: {image_size}")
        
        # サイズが一致しない場合はスキップ
        if img.shape != image_size:
            print(f"Warning: {image_path.name} のサイズが異なります。スキップします。")
            continue

        match = wavelength_pattern.search(image_path.name)
        if not match:
            continue
        ex_wavelength = int(match.group(1))
        em_wavelength = int(match.group(2))
        
        pixel_features_df[f'Ex{ex_wavelength}_Em{em_wavelength}'] = img.flatten()
        processed_files_count += 1
    except Exception as e:
        print(f"Warning: Failed to process {image_path.name}. Reason: {e}")
        continue

if processed_files_count > 0:
    # 3. 正解ラベルの列をデータフレームに追加
    # 動的に取得した画像サイズでマスクを生成
    pixel_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)
    
    # マスクを1次元配列に変換
    pixel_labels_flat = pixel_label_mask.flatten()

    pixel_features_df['label_value'] = pixel_labels_flat
    pixel_features_df['label_name'] = pd.Series(pixel_labels_flat).map({v: k for k, v in label_name_to_value.items()})

    # データセットをCSVとして保存
    output_csv_path = main_dir / "pixel_features_all_data.csv"
    pixel_features_df.to_csv(output_csv_path, index=False)
    
    print(f'\nピクセル単位のデータセットが作成されました: {output_csv_path}')
    print('\nデータセットのプレビュー:')
    print(pixel_features_df.head())
else:
    print("Warning: No data was successfully processed. Check your files and paths.")

# %% [markdown]
# ## 学習と波長選択(散乱光を除く)

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 作成されたピクセル単位のデータセットを読み込む
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
output_csv_path = main_dir / "pixel_features_all_data.csv"

try:
    df_pixels = pd.read_csv(output_csv_path)
except FileNotFoundError:
    print(f"Error: {output_csv_path} が見つかりません。")
    exit()

# 散乱光の波長を除外する処理
# 列名から'Ex'と'Em'の波長が同じものをフィルタリング
scatter_light_cols = [col for col in df_pixels.columns if 'Ex' in col and 'Em' in col and col.split('_')[0].replace('Ex', '') == col.split('_')[1].replace('Em', '')]
df_filtered = df_pixels.drop(columns=scatter_light_cols)

df_labeled_only = df_filtered[df_filtered['label_name'] != '_background_']

if df_labeled_only.empty:
    print("Warning: データセットにラベル付けされたプラスチックのピクセルがありません。")
    exit()

X = df_labeled_only.drop(columns=['label_value', 'label_name'])
y = df_labeled_only['label_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_save_path = main_dir / "random_forest_model_no_scatter.joblib"
joblib.dump(model, model_save_path)
print(f"\n学習済みモデルを保存しました: {model_save_path}")

y_pred = model.predict(X_test)
print('--- 精度評価レポート（散乱光除去後） ---')
print(classification_report(y_test, y_pred))

feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)
print("\n特徴量の重要度ランキング（トップ10）:")
print(importance_df.head(10))

# %% [markdown]
# ---
# # 選択した波長の分光画像からの画像分類

# %% [markdown]
# ## データセットの作成


