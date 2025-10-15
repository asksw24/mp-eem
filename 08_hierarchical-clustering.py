# %% [markdown]
# # 階層型クラスタリング（平均スペクトル）

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
csv_filename = "pixel_features_plastics_only.csv"
# --------------------

# --- ステップ1: 平均スペクトルデータの作成 ---
print("--- ステップ1: 平均スペクトルデータの作成 ---")

# 1. 2つのデータセットを読み込み、結合する
try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print("データセットの結合が完了しました。")
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません: {e.filename}")
    exit()

# 2. 特徴量とラベルに分割
X_full = combined_df.drop(columns=['label_name', 'original_index'])
y_full = combined_df['label_name']

# 3. 各プラスチックの平均スペクトルを計算
mean_spectra_df = pd.concat([X_full, y_full], axis=1).groupby('label_name').mean()
print("各プラスチックの平均スペクトルを計算しました。")

# --- ステップ2: コサイン類似度行列の作成 ---
print("\n--- ステップ2: コサイン類似度行列の作成 ---")

# コサイン類似度を計算 (値が1に近いほど似ている)
similarity_matrix = cosine_similarity(mean_spectra_df)
similarity_df = pd.DataFrame(similarity_matrix, index=mean_spectra_df.index, columns=mean_spectra_df.index)
print("コサイン類似度行列:")
print(similarity_df)

# --- ステップ3: 階層的クラスタリングの実行 ---
print("\n--- ステップ3: 階層的クラスタリングの実行 ---")

# 類似度(similarity)を距離(distance)に変換
distance_matrix = 1 - similarity_matrix

# 浮動小数点誤差を補正するため、対角成分を強制的に0にする
np.fill_diagonal(distance_matrix, 0)

condensed_distance = squareform(distance_matrix)

# 階層的クラスタリングを実行
linked = hierarchy.linkage(condensed_distance, method='ward')
print("階層的クラスタリングが完了しました。")


# --- ステップ4: デンドログラムの可視化 ---
print("\n--- ステップ4: デンドログラムの可視化 ---")
# ★★★ スタイルシート名を新しいバージョン対応のものに変更 ★★★
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# デンドログラムを描画
hierarchy.dendrogram(
    linked,
    orientation='top',
    labels=mean_spectra_df.index,
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Hierarchical Clustering Dendrogram of Plastics', fontsize=16)
plt.ylabel('Distance (1 - Cosine Similarity)', fontsize=12)
plt.xlabel('Plastic Type', fontsize=12)
plt.tight_layout()
plt.show()



# %% [markdown]
# 種類別生産割合の追加

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
csv_filename = "pixel_features_plastics_only.csv"
# --------------------

# --- ステップ1: 平均スペクトルデータの作成 ---
print("--- ステップ1: 平均スペクトルデータの作成 ---")

# 1. 2つのデータセットを読み込み、結合する
try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print("データセットの結合が完了しました。")
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません: {e.filename}")
    exit()

# 2. 特徴量とラベルに分割
X_full = combined_df.drop(columns=['label_name', 'original_index'])
y_full = combined_df['label_name']

# 3. 各プラスチックの平均スペクトルを計算
mean_spectra_df = pd.concat([X_full, y_full], axis=1).groupby('label_name').mean()
print("各プラスチックの平均スペクトルを計算しました。")

# --- ステップ2: コサイン類似度行列の作成 ---
print("\n--- ステップ2: コサイン類似度行列の作成 ---")
similarity_matrix = cosine_similarity(mean_spectra_df)

# --- ステップ3: 階層的クラスタリングの実行 ---
print("\n--- ステップ3: 階層的クラスタリングの実行 ---")
distance_matrix = 1 - similarity_matrix
np.fill_diagonal(distance_matrix, 0)
condensed_distance = squareform(distance_matrix)
linked = hierarchy.linkage(condensed_distance, method='ward')
print("階層的クラスタリングが完了しました。")

# --- ステップ4: デンドログラムの可視化 ---
print("\n--- ステップ4: デンドログラムの可視化 ---")

# ★★★ ここからが変更点 ★★★
# 1. 生産割合のデータを定義 (ご提示のグラフより)
production_ratios = {
    'PP': 19.0,
    'LDPE': 14.0, # グラフでは LD-PE
    'PVC': 12.8,
    'HDPE': 12.2, # グラフでは HD-PE
    'PET': 6.2,
    'PS': 5.2,
    'PC': 1.9,   # PC, PMMA, ABSはグラフにないため不明
    'PMMA': 0.8,
    'ABS': 2.4
}

# 2. デンドログラムのラベルを「名前 + 生産割合」の形式に動的に作成
original_labels = mean_spectra_df.index
new_labels = []
for label in original_labels:
    ratio = production_ratios.get(label)
    if pd.notna(ratio):
        new_labels.append(f"{label} ({ratio}%)")
    else:
        new_labels.append(f"{label} (N/A)")
# ★★★ ここまでが変更点 ★★★


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 8))

# デンドログラムを描画 (★★★ labelsに新しいラベルリストを使用 ★★★)
hierarchy.dendrogram(
    linked,
    orientation='top',
    labels=new_labels,
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Hierarchical Clustering Dendrogram of Plastics (with Production Ratios)', fontsize=16)
plt.ylabel('Distance (1 - Cosine Similarity)', fontsize=12)
plt.xlabel('Plastic Type (Production Ratio %)', fontsize=12)
plt.tight_layout()
plt.show()



# %% [markdown]
# ---

# %% [markdown]
# ## インスタンス（プラスチック片）単位

# %% [markdown]
# ### Baseline：通常の9クラス分類

# %%
import pandas as pd
import numpy as np
import labelme
import json
from PIL import Image
from pathlib import Path
import re
from skimage import measure # 連結成分ラベリングに使用

# --- ユーザー設定 ---
# このスクリプトは、2つのデータセットフォルダそれぞれに対して実行する必要があります
folder_name = "MPs_20250911" # まずはこちらで実行
# folder_name = "MPs_20250905_2" # 次にこちらで実行
main_dir = Path(f"C:/Users/sawamoto24/sawamoto24/master/microplastic/data/{folder_name}")
reference_file_stem = f"{folder_name}_Ex-1_Em-1_ET300_step1"
# --------------------


# --- ステップ1: インスタンスマスクの作成 ---
print(f"--- {folder_name}: インスタンスID付きデータセットの作成 ---")
json_path = main_dir / (reference_file_stem + ".json")
with open(json_path, 'r') as f:
    data = json.load(f)

image_size = (data['imageHeight'], data['imageWidth'])
labels_in_json = sorted(list(set(shape['label'] for shape in data['shapes'])))
label_name_to_value = {label: i for i, label in enumerate(labels_in_json, start=1)}

class_label_mask, _ = labelme.utils.shapes_to_label(image_size, data['shapes'], label_name_to_value)

# 連結成分ラベリングで各プラスチック片にユニークIDを割り振る
instance_mask = np.zeros_like(class_label_mask, dtype=int)
instance_id_counter = 1
for label_name, label_value in label_name_to_value.items():
    if label_name not in ['other', 'background', 'background_ref']:
        binary_mask = (class_label_mask == label_value)
        # 各連結成分（個々の物体）にIDを振る
        labeled_components, num_components = measure.label(binary_mask, connectivity=2, return_num=True)
        for i in range(1, num_components + 1):
            instance_mask[labeled_components == i] = instance_id_counter
            instance_id_counter += 1
print(f"合計 {instance_id_counter - 1} 個のプラスチック片（インスタンス）を検出しました。")

# --- ステップ2: スペクトルデータとラベルの結合 ---
# (この部分は以前のスクリプトとほぼ同じ)
wavelength_pattern = re.compile(r'Ex(\d+)_Em(\d+)')
image_files = list(main_dir.glob("*.tiff"))
pixel_features_df = pd.DataFrame()

for image_path in image_files:
    if '-1_Em-1' in image_path.stem: continue
    match = wavelength_pattern.search(image_path.name)
    if not match: continue
    ex, em = int(match.group(1)), int(match.group(2))
    if ex == em: continue
    try:
        img = np.asarray(Image.open(image_path))
        if img.shape[:2] != image_size: continue
        pixel_features_df[f'Ex{ex}_Em{em}'] = img.flatten()
    except Exception as e:
        print(f"Warning: Failed to process {image_path.name}. Reason: {e}")

# --- ステップ3: 新しい列を追加して整形 ---
print("\nスペクトルデータにラベルとIDを追加します...")
pixel_features_df.reset_index(inplace=True)
pixel_features_df.rename(columns={'index': 'original_index'}, inplace=True)

# ラベル名をマッピング
value_to_label_name = {v: k for k, v in label_name_to_value.items()}
value_to_label_name[0] = '_unlabeled_' # ラベルなし領域
pixel_features_df['label_name'] = pd.Series(class_label_mask.flatten()).map(value_to_label_name)

# ★★★ インスタンスID列を追加 ★★★
pixel_features_df['instance_id'] = instance_mask.flatten()

# 不要なピクセルを除外（プラスチックのみを残す）
labels_to_exclude = ['other', 'background', 'background_ref', '_unlabeled_']
plastics_only_df = pixel_features_df[~pixel_features_df['label_name'].isin(labels_to_exclude)].copy()
print(f"プラスチックのピクセルのみを抽出しました。総ピクセル数: {len(plastics_only_df)}")

# --- ステップ4: CSVとして保存 ---
output_dir = main_dir / "csv"
output_dir.mkdir(parents=True, exist_ok=True)
# ★★★ 新しいファイル名で保存 ★★★
output_csv_path = output_dir / "pixel_features_with_instance_id.csv"
plastics_only_df.to_csv(output_csv_path, index=False)

print(f'\nインスタンスID付きの新しいデータセットが作成されました: {output_csv_path}')
print(plastics_only_df.head())

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
csv_filename = "pixel_features_with_instance_id.csv"
# --------------------

# --- ステップ1: 2つのデータセットを読み込み、結合 ---
print("--- ステップ1: インスタンスID付きデータセットの結合 ---")
try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
    
    # instance_idが重複しないように、データセット2のIDにオフセットを追加
    df2['instance_id'] = df2['instance_id'] + 1000 
    
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print("データセットの結合が完了しました。")
    print(f"総インスタンス数: {combined_df['instance_id'].nunique()} 個")

except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません: {e.filename}")
    print("両方のデータセットで「インスタンスID付きデータセットの作成」スクリプトを実行したか確認してください。")
    exit()

# --- ステップ2: データ準備 ---
print("\n--- ステップ2: データ準備 ---")
X = combined_df.drop(columns=['label_name', 'original_index', 'instance_id'])
y = combined_df['label_name']
groups = combined_df['instance_id']

# --- ステップ3: リーブ・ワン・アウト交差検証 (インスタンス単位) ---
print("\n--- ステップ3: インスタンス単位のリーブ・ワン・アウト交差検証を開始 ---")

logo = LeaveOneGroupOut()
all_y_pred = []
all_y_true = []
num_splits = logo.get_n_splits(groups=groups)
current_split = 0

for train_idx, test_idx in logo.split(X, y, groups):
    current_split += 1
    print(f"検証中... {current_split}/{num_splits}")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    all_y_pred.extend(y_pred)
    all_y_true.extend(y_test)

# --- ステップ4: 最終的な精度レポートの作成 ---
print("\n--- ステップ4: 最終的な精度レポート ---")
report_str = classification_report(all_y_true, all_y_pred)
print("インスタンス単位での交差検証が完了しました。")
print(report_str)

# --- ステップ5: 予測結果をファイルに保存 ---
results_df = pd.DataFrame({
    'true_label': all_y_true,
    'predicted_label': all_y_pred
})
output_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/results/インスタンス単位検証")
output_dir.mkdir(parents=True, exist_ok=True)
results_csv_path = output_dir / "prediction_results_loocv.csv"
results_df.to_csv(results_csv_path, index=False)

report_output_path = output_dir / "classification_report_instance_level_loocv.txt"
with open(report_output_path, 'w', encoding='utf-8') as f:
    f.write("--- インスタンス単位 リーブ・ワン・アウト交差検証 精度レポート ---\n\n")
    f.write(report_str)

print(f"\n予測結果とレポートを {output_dir} に保存しました。")



# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ユーザー設定 ---
# 検証結果が保存されているCSVファイルのパス
results_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/results/インスタンス単位検証")
results_csv_path = results_dir / "prediction_results_loocv.csv"
# --------------------

# 1. 予測結果のCSVファイルを読み込む
print(f"予測結果ファイルを読み込みます: {results_csv_path}")
try:
    results_df = pd.read_csv(results_csv_path)
    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']
except FileNotFoundError:
    print(f"エラー: 予測結果ファイルが見つかりません。")
    print("先に「インスタンス単位でのリーブ・ワン・アウト交差検証」スクリプトを実行してください。")
    exit()

# 2. 混同行列の計算
print("\n--- 混同行列の計算 ---")
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 3. 混同行列の可視化 (生データ)
print("\n--- 混同行列の可視化 (絶対数) ---")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Instance-Level LOOCV - Raw Counts)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# 4. 正規化された混同行列の可視化 (割合)
print("\n--- 正規化された混同行列の可視化 (Recall) ---")
cm_sum = cm.sum(axis=1)[:, np.newaxis]
with np.errstate(divide='ignore', invalid='ignore'):
    cm_normalized = np.nan_to_num(cm.astype('float') / cm_sum)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Normalized Confusion Matrix (Instance-Level LOOCV - Recall)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()



# %% [markdown]
# ### クラス別分類

# %% [markdown]
# 階層別にデータセットを作成

# %%
import pandas as pd
from pathlib import Path

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
csv_filename = "pixel_features_with_instance_id.csv"

# デンドログラムに基づいた階層（サブクラスタ）を定義
level1_map = {
    'PVC': 'Group_A', 'PS': 'Group_A', 'ABS': 'Group_A', 'PET': 'Group_A',
    'PMMA': 'Group_B', 'PC': 'Group_B', 'PP': 'Group_B', 'LDPE': 'Group_B', 'HDPE': 'Group_B'
}
level2_map = {
    'PVC': 'Subgroup_A1 (PVC/PS)', 'PS': 'Subgroup_A1 (PVC/PS)', 
    'ABS': 'Subgroup_A2 (ABS)', 
    'PET': 'Subgroup_A3 (PET)'
}
level3_map = {
    'PP': 'Subgroup_B1 (PP/LDPE)', 'LDPE': 'Subgroup_B1 (PP/LDPE)', 
    'HDPE': 'Subgroup_B2 (HDPE)', 
    'PC': 'Subgroup_B3 (PC/PMMA)', 'PMMA': 'Subgroup_B3 (PC/PMMA)'
}
# --------------------

# 1. データの読み込みと結合
print("--- ステップ1: 階層ラベルデータセットの作成 ---")
try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
    
    # instance_idが重複しないようにオフセットを追加
    df2['instance_id'] = df2['instance_id'] + 1000 
    
    combined_df_hierarchical = pd.concat([df1, df2], ignore_index=True)
    print("データセットの結合が完了しました。")

except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません: {e.filename}")
    print("両方のデータセットで「インスタンスID付きデータセットの作成」スクリプトを実行したか確認してください。")
    # exit() # ipynbではexit()をコメントアウトした方が安全です

# 2. 新しい階層ラベル列を追加
combined_df_hierarchical['level1'] = combined_df_hierarchical['label_name'].map(level1_map)
combined_df_hierarchical['level2'] = combined_df_hierarchical['label_name'].map(level2_map)
combined_df_hierarchical['level3'] = combined_df_hierarchical['label_name'].map(level3_map)

print("階層ラベルの追加が完了しました。")
print("データセットのプレビュー:")
print(combined_df_hierarchical[['label_name', 'level1', 'level2', 'level3']].head())

# %% [markdown]
# モデルの学習と検証

# %%
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path

# --- ユーザー設定 ---
# 評価したい階層レベルをリストで指定
target_levels = ['level1', 'level2', 'level3', 'label_name'] 
# 結果を保存するフォルダを定義
output_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/results/階層別分類_インスタンス単位検証")
# --------------------

# 出力フォルダの作成
output_dir.mkdir(parents=True, exist_ok=True)
print(f"結果は {output_dir} に保存されます。")

# --- ループ処理で各レベルを評価 ---
for target_level in target_levels:
    print(f"\n{'='*60}")
    print(f"--- {target_level} レベルでのインスタンス単位LOOCVを開始 ---")

    # 1. データ準備
    # NaNが含まれる行（その階層に属さないクラス）を除外
    level_df = combined_df_hierarchical.dropna(subset=[target_level])

    if level_df.empty:
        print(f"{target_level} に該当するデータがありません。スキップします。")
        continue

    # 特徴量 (X), ラベル (y), グループ (instance_id) を準備
    X = level_df.drop(columns=['label_name', 'original_index', 'instance_id', 'level1', 'level2', 'level3'])
    y = level_df[target_level]
    groups = level_df['instance_id']

    # データが1クラスしかない場合は分割できないためスキップ
    if y.nunique() < 2:
        print(f"{target_level} のクラスが1種類しかないため、評価をスキップします。")
        continue

    # 2. インスタンス単位のリーブ・ワン・アウト交差検証
    logo = LeaveOneGroupOut()
    all_y_pred = []
    all_y_true = []
    
    num_splits = logo.get_n_splits(groups=groups)
    current_split = 0

    for train_idx, test_idx in logo.split(X, y, groups):
        current_split += 1
        print(f"検証中... {current_split}/{num_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        all_y_pred.extend(y_pred)
        all_y_true.extend(y_test)

    # 3. 最終的な精度レポートの作成と保存
    report_str = classification_report(all_y_true, all_y_pred, zero_division=0)

    print(f"\n--- {target_level} 精度評価レポート ---")
    print(report_str)

    output_path = output_dir / f"classification_report_{target_level}_instance_level.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"--- {target_level} | インスタンス単位LOOCV 精度レポート ---\n\n")
        f.write(report_str)
    print(f"レポートを {output_path} に保存しました。")

    # 4. 特徴量重要度の算出と保存
    print(f"\n--- {target_level} レベルの特徴量重要度を計算 ---")
    # このレベルの全データを使って最終モデルを一度だけ学習
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    final_model.fit(X, y)

    # 重要度をDataFrameに変換
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # CSVファイルとして保存
    importance_output_path = output_dir / f"feature_importance_{target_level}.csv"
    importances_df.to_csv(importance_output_path, index=False)
    print(f"特徴量重要度を {importance_output_path} に保存しました。")

print(f"\n{'='*60}")
print("全ての階層レベルの評価が完了しました。")



# %% [markdown]
# ---
# ## 階層的分類のパイプライン検証

# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- ユーザー設定 ---
main_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/data")
dataset1_folder_name = "MPs_20250911"
dataset2_folder_name = "MPs_20250905_2"
csv_filename = "pixel_features_with_instance_id.csv"
output_dir = Path("C:/Users/sawamoto24/sawamoto24/master/microplastic/results/階層的分類_パイプライン検証_LOOCV")
# --------------------

# --- ステップ1: 階層ラベル付きデータセットの作成 ---
print("--- ステップ1: 階層ラベルデータセットの作成 ---")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"結果は {output_dir} に保存されます。")

level1_map = {
    'PVC': 'Group_A', 'PS': 'Group_A', 'ABS': 'Group_A', 'PET': 'Group_A',
    'PMMA': 'Group_B', 'PC': 'Group_B', 'PP': 'Group_B', 'LDPE': 'Group_B', 'HDPE': 'Group_B'
}

try:
    df1 = pd.read_csv(main_dir / dataset1_folder_name / "csv" / csv_filename)
    df2 = pd.read_csv(main_dir / dataset2_folder_name / "csv" / csv_filename)
    df2['instance_id'] = df2['instance_id'] + 1000 
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df['level1'] = combined_df['label_name'].map(level1_map)
    print("データセットの準備が完了しました。")
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません: {e.filename}")
    exit()

# --- ステップ2: インスタンス単位のリーブ・ワン・アウト交差検証 ---
print("\n--- ステップ2: インスタンス単位での検証を開始 ---")
X = combined_df.drop(columns=['label_name', 'original_index', 'instance_id', 'level1'])
y = combined_df['label_name']
y_level1 = combined_df['level1']
groups = combined_df['instance_id']

logo = LeaveOneGroupOut()
pipeline_preds, baseline_preds, all_true = [], [], []
num_splits = logo.get_n_splits(groups=groups)
current_split = 0

for train_idx, test_idx in logo.split(X, y, groups):
    current_split += 1
    print(f"検証中... {current_split}/{num_splits}")

    # --- ★★★ ループ内で毎回、訓練データとテストデータを分割 ★★★ ---
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_level1_train = y_level1.iloc[train_idx]
    
    # --- ★★★ ループ内で毎回、モデルをゼロから学習 ★★★ ---
    # Level 1 モデル (Group A vs B)
    model_level1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1).fit(X_train, y_level1_train)

    # Level 2 モデル (Group A内での分類)
    df_group_a_train = combined_df.iloc[train_idx][combined_df.iloc[train_idx]['level1'] == 'Group_A']
    X_group_a_train = df_group_a_train.drop(columns=['label_name', 'original_index', 'instance_id', 'level1'])
    y_group_a_train = df_group_a_train['label_name']
    model_level2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1).fit(X_group_a_train, y_group_a_train)
    
    # Level 3 モデル (Group B内での分類)
    df_group_b_train = combined_df.iloc[train_idx][combined_df.iloc[train_idx]['level1'] == 'Group_B']
    X_group_b_train = df_group_b_train.drop(columns=['label_name', 'original_index', 'instance_id', 'level1'])
    y_group_b_train = df_group_b_train['label_name']
    model_level3 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1).fit(X_group_b_train, y_group_b_train)
    
    # ベースラインモデル (通常の9クラス分類)
    model_baseline = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1).fit(X_train, y_train)

    # --- ★★★ 学習させたモデルでテストデータを予測 ★★★ ---
    # 階層パイプライン
    level1_pred = model_level1.predict(X_test)
    final_pred = np.empty_like(level1_pred)
    is_group_a = (level1_pred == 'Group_A')
    is_group_b = (level1_pred == 'Group_B')
    if np.any(is_group_a): final_pred[is_group_a] = model_level2.predict(X_test[is_group_a])
    if np.any(is_group_b): final_pred[is_group_b] = model_level3.predict(X_test[is_group_b])
    pipeline_preds.extend(final_pred)
    
    # ベースラインモデル
    baseline_preds.extend(model_baseline.predict(X_test))
    
    all_true.extend(y_test)

# --- ステップ3: 結果の比較と保存 ---
print("\n--- ステップ3: 結果の比較と保存 ---")

# ベースラインモデル
baseline_report_str = classification_report(all_true, baseline_preds)
print("\n" + "="*20 + " ベースラインモデル (9クラス) " + "="*20)
print(baseline_report_str)
with open(output_dir / "report_baseline_model.txt", 'w', encoding='utf-8') as f:
    f.write(baseline_report_str)

# 階層的分類パイプライン
pipeline_report_str = classification_report(all_true, pipeline_preds)
print("\n" + "="*20 + " 階層的分類パイプライン " + "="*20)
print(pipeline_report_str)
with open(output_dir / "report_pipeline_model.txt", 'w', encoding='utf-8') as f:
    f.write(pipeline_report_str)

print(f"\n全てのレポートを {output_dir} に保存しました。")




