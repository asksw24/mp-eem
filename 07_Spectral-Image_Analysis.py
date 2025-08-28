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

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re

# --- ユーザーが設定する項目 ---
# 1. すべての画像とJSONファイルが入っているフォルダのパス
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
# 2. ラベル付けを行った基準画像のファイル名（拡張子なし）
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

# ラベル名と数値の対応辞書を一度だけ作成
labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0

# 結果を保存するリストを初期化
all_results = []

# ファイル名からExとEmの波長を抽出する新しい正規表現
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')

# 2. フォルダ内の分光画像ファイルだけを対象にループ処理
#    ファイル名に"Ex"と"Em"が含まれるtiffファイルを対象とします
image_files = list(main_dir.glob("*Ex*_Em*.tiff"))
# 基準画像（Em-1）は除外
image_files = [f for f in image_files if '-1-Em-1-' not in f.stem]

if not image_files:
    print("Error: 指定されたパターンに一致する分光画像ファイルが見つかりません。")
    exit()

print(f"\nFound {len(image_files)} spectral image files to process.")

for image_path in image_files:
    print(f"\nProcessing image: {image_path.name}")

    try:
        # ファイル名から励起波長(Ex)と放射波長(Em)を抽出
        match = wavelength_pattern.search(image_path.name)
        if not match:
            print(f"Warning: Wavelength pattern not found in {image_path.name}. Skipping.")
            continue
        
        ex_wavelength = int(match.group(1))
        em_wavelength = int(match.group(2))

        # 画像を読み込む
        img = np.asarray(Image.open(image_path))

        # マスクを生成
        lbl, _ = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        
        # 各ラベルごとに画素値を抽出し、統計量を算出
        for label_name, value in label_name_to_value.items():
            if label_name == '_background_':
                continue
            
            label_mask = (lbl == value)
            pixel_values = img[label_mask]

            if len(pixel_values) > 0:
                result = {
                    'image_name': image_path.name,
                    'label': label_name,
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
    output_csv_path = main_dir / "combined_spectral_features.csv"
    df.to_csv(output_csv_path, index=False)
    print(f'\n全分光データの統合データセットが作成されました: {output_csv_path}')
    print('\nデータセットのプレビュー:')
    print(df.head())
else:
    print("Warning: No data was successfully processed. Check your files and paths.")

# %%
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 作成された統合データセットのパスを指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
output_csv_path = main_dir / "combined_spectral_features.csv"

# データセットを読み込み
try:
    df = pd.read_csv(output_csv_path)
except FileNotFoundError:
    print(f"Error: {output_csv_path} が見つかりません。")
    exit()

# 特徴量（X）とラベル（y）に分割
# 特徴量には、波長情報と統計量を含めます
X = df.drop(['image_name', 'label'], axis=1)
# ラベルは 'label' 列
y = df['label']

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# サポートベクターマシン（SVM）モデルを初期化
model = SVC()

# モデルを学習
model.fit(X_train, y_train)

print('機械学習モデルの学習が完了しました。')

# %%
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 作成された統合データセットのパスを指定
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
output_csv_path = main_dir / "combined_spectral_features.csv"

# データセットを読み込み
try:
    df = pd.read_csv(output_csv_path)
except FileNotFoundError:
    print(f"Error: {output_csv_path} が見つかりません。")
    exit()

# 特徴量（X）とラベル（y）に分割
X = df.drop(['image_name', 'label'], axis=1)
y = df['label']

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# サポートベクターマシン（SVM）モデルを初期化し、学習
model = SVC()
model.fit(X_train, y_train)

# テストデータに対する予測を実行
y_pred = model.predict(X_test)

# 精度評価レポートを表示
print('--- 精度評価レポート ---')
print(classification_report(y_test, y_pred))

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from pathlib import Path
import labelme
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import re

# --- ユーザーが設定する項目 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data/MPs_15cm_20250826")
# 評価したい分光画像のファイル名（拡張子なし）
evaluate_file_stem = "MPs_15cm_20250826_Ex260_Em280_ET20000_step1"
# JSONファイルのパス
json_path = main_dir / "MPs_15cm_20250826_Ex-1_Em-1_ET300_step1.json"
# ------------------------------

# 1. データセットとモデルの準備
df = pd.read_csv(main_dir / "combined_spectral_features.csv")
X = df.drop(['image_name', 'label'], axis=1)
y = df['label']
model = SVC()
model.fit(X, y)

# 2. 評価したい画像の処理
with open(json_path, 'r') as f:
    data = json.load(f)
labels = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels, start=1)}
label_name_to_value['_background_'] = 0

image_path = main_dir / (evaluate_file_stem + ".tiff")
img = np.asarray(Image.open(image_path))
lbl, _ = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

# 3. 各ラベル領域の画素値からモデルで予測
unique_labels = sorted([l for l in np.unique(lbl) if l > 0])
label_value_to_predicted_name = {}

# ファイル名から波長情報を抽出
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
match = wavelength_pattern.search(evaluate_file_stem)
ex_wavelength = int(match.group(1))
em_wavelength = int(match.group(2))

for label_value in unique_labels:
    pixel_values = img[lbl == label_value]
    if len(pixel_values) > 0:
        feature_vector = {
            'Ex_wavelength': ex_wavelength,
            'Em_wavelength': em_wavelength,
            'pixel_count': len(pixel_values),
            'mean': np.mean(pixel_values),
            'std_dev': np.std(pixel_values),
            'max_value': np.max(pixel_values),
            'min_value': np.min(pixel_values)
        }
        single_df = pd.DataFrame([feature_vector])
        predicted_class = model.predict(single_df)[0]
        label_value_to_predicted_name[label_value] = predicted_class
    
# 4. 予測結果の可視化
# 予測ラベル（文字列）を数値に変換する辞書を動的に作成
predicted_labels = list(set(label_value_to_predicted_name.values()))
predicted_label_to_color_index = {label: i for i, label in enumerate(predicted_labels)}

# 予測結果の数値マスクを生成
colored_mask = np.zeros(img.shape, dtype=int)
for label_value, predicted_name in label_value_to_predicted_name.items():
    color_index = predicted_label_to_color_index[predicted_name]
    colored_mask[lbl == label_value] = color_index

# カラーマップを生成
colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
    '#00FFFF', '#FFA500', '#800080', '#A52A2A', '#000000' # 黒は背景として追加
]
cmap = ListedColormap(colors[:len(predicted_labels)])

plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
plt.imshow(colored_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=len(predicted_labels)-1)
plt.title(f'Predicted Plastic Types for {evaluate_file_stem}')
plt.colorbar(ticks=range(len(predicted_labels)), label='Plastic Type')
plt.show()

# 予測結果を表示
print("\nPredicted Labels:")
for label_value, predicted_name in label_value_to_predicted_name.items():
    print(f"Original Label Value: {label_value} -> Predicted: {predicted_name}")


