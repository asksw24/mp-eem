# %% [markdown]
# # labelmeの結果可視化

# %%
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path

# プロジェクトのメインディレクトリとファイル名を指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1"

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
import labelme
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path

# プロジェクトのメインディレクトリ
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")

# 画像とJSONファイルのパス
json_path = main_dir / "MPs_20250905_2_Ex-1_Em-1_ET300_step1.json"
image_path = main_dir / "MPs_20250905_2_Ex-1_Em-1_ET300_step1.tiff"
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
spectral_image_path = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2/MPs_20250905_2_Ex360_Em480_ET10000_step1.tiff")

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
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
reference_file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）
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
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
# reference_file_stem = "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1" # ラベリングに使用した画像ファイル名（拡張子なし）
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
image_files = list(main_dir.glob("*.tiff")) # フィルタなし画像もglobで取得

if not image_files:
    print("Error: 指定されたパターンに一致する分光画像ファイルが見つかりません。")
    exit()

pixel_features_df = pd.DataFrame()
processed_files_count = 0
image_size = None

for image_path in image_files:
    # フィルタなしの画像を除外
    if '-1_Em-1' in image_path.stem:
        print(f"Skipping filter-less image: {image_path.name}")
        continue
    
    match = wavelength_pattern.search(image_path.name)
    if not match:
        continue

    ex_wavelength = int(match.group(1))
    em_wavelength = int(match.group(2))

    # 1次散乱光のデータを除外
    if ex_wavelength == em_wavelength:
        print(f"Skipping primary scattered light: {image_path.name}")
        continue

    try:
        img = np.asarray(Image.open(image_path))
        if image_size is None:
            image_size = img.shape
            print(f"画像サイズを検出しました: {image_size}")
        
        if img.shape != image_size:
            print(f"Warning: {image_path.name} のサイズが異なります。スキップします。")
            continue
        
        pixel_features_df[f'Ex{ex_wavelength}_Em{em_wavelength}'] = img.flatten()
        processed_files_count += 1
    except Exception as e:
        print(f"Warning: Failed to process {image_path.name}. Reason: {e}")
        continue

if processed_files_count > 0:
    # 3. 正解ラベルの列をデータフレームに追加
    pixel_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)
    
    pixel_labels_flat = pixel_label_mask.flatten()

    pixel_features_df['label_value'] = pixel_labels_flat
    pixel_features_df['label_name'] = pd.Series(pixel_labels_flat).map({v: k for k, v in label_name_to_value.items()})

    # --- 背景を除外 ---
    before_rows = len(pixel_features_df)
    pixel_features_df = pixel_features_df[pixel_features_df['label_name'] != '_background_'].reset_index(drop=True)
    after_rows = len(pixel_features_df)
    print(f"背景を除外しました: {before_rows - after_rows} 行削除, 残り {after_rows} 行")

    # データセットをCSVとして保存
    output_csv_path = main_dir / "pixel_features_no_background.csv"
    pixel_features_df.to_csv(output_csv_path, index=False)
    
    print(f'\nピクセル単位のデータセットが作成されました: {output_csv_path}')
    print('\nデータセットのプレビュー:')
    print(pixel_features_df.head())
else:
    print("Warning: No data was successfully processed. Check your files and paths.")


# %% [markdown]
# ## 学習と波長選択(1次散乱光、フィルタなし画像を除く)

# %%
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# # --- ユーザーが設定する項目 ---
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
# # ------------------------------

# 作成されたピクセル単位のデータセットを読み込む
output_csv_path = main_dir / "pixel_features_all_data.csv"

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
importance_csv_path = main_dir / "csv" / "selected_wavelengths_importance.csv"
# 親ディレクトリが存在しない場合は作成
importance_csv_path.parent.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(importance_csv_path, index=False)

print(f"\n特徴量重要度ランキングを保存しました: {importance_csv_path}")
print("\n上位5件の特徴量:")
print(importance_df.head())


# %%
print("\n特徴量の重要度ランキング（トップ10）:")
print(importance_df.head(10))

# %% [markdown]
# ---
# # 選択した波長の分光画像からの画像分類

# %% [markdown]
# ## データセットの作成

# %%
import pandas as pd
from pathlib import Path

# # --- ユーザーが設定する項目 ---
# main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
n_features = 10  # 上位何個の波長を使うか指定

# 1. 特徴量重要度ランキングを読み込み
importance_csv_path = main_dir / "csv" / "selected_wavelengths_importance.csv"
importance_df = pd.read_csv(importance_csv_path)

# 上位 n_features の波長名を取得
top_features = importance_df['feature'].head(n_features).tolist()
print(f"選択された上位 {n_features} 波長: {top_features}")

# 2. ピクセル単位データを読み込み
pixel_features_all_data_path = main_dir / "pixel_features_all_data.csv"
df_pixels = pd.read_csv(pixel_features_all_data_path)

# 背景を除外（ラベル名が _background_ の行を削除）
df_labeled_only = df_pixels[df_pixels['label_name'] != '_background_']

# 3. 必要な列だけ抽出
df_selected = df_labeled_only[['label_value', 'label_name'] + top_features]

# 4. 保存
output_csv_path = main_dir / f"pixel_features_top{n_features}_wavelengths.csv"
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
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
reference_file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1"   # 元画像（JSON・tiff）の基準
visualize_file_stem = "MPs_20250905_2_Ex260_Em280_ET10000_step1"  # 可視化対象画像

top_features = [
    'Ex360_Em380', 'Ex320_Em360', 'Ex380_Em420', 'Ex260_Em360', 'Ex300_Em380',
    'Ex360_Em460', 'Ex300_Em360', 'Ex280_Em380', 'Ex340_Em380', 'Ex340_Em400'
]
# --------------------

# 1. データ読み込み
pixel_features_all_data_path = main_dir / "pixel_features_all_data.csv"
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

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_20250905_2")
reference_file_stem = "MPs_20250905_2_Ex-1_Em-1_ET300_step1"   # 元画像（JSON・tiff）の基準
visualize_file_stem = "MPs_20250905_2_Ex260_Em280_ET10000_step1"  # 可視化対象画像
top_features = [
    'Ex360_Em380', 'Ex320_Em360', 'Ex380_Em420', 'Ex260_Em360', 'Ex300_Em380',
    'Ex360_Em460', 'Ex300_Em360', 'Ex280_Em380', 'Ex340_Em380', 'Ex340_Em400'
]
# --------------------

# 1. データ読み込み
pixel_features_all_data_path = main_dir / "pixel_features_all_data.csv"
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


