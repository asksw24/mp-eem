# %% [markdown]
# # Make Camera Spectral Sensitivity

# %%
import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
wl_start = 200  # nm
wl_end = 1000   # nm
wl_step = 1     # 分解能
wavelengths = np.arange(wl_start, wl_end + 1, wl_step)

# チャネルごとの設定
channels = [
    {"peak": 500, "fwhm": 380, "max_val": 1.0},
    {"peak": 230, "fwhm": 40, "max_val": 0.3}
]

# 感度行列（列ごとにチャネル）
sensitivity_all = []

for ch in channels:
    sigma = ch["fwhm"] / 2.355
    sensitivity = np.exp(-0.5 * ((wavelengths - ch["peak"]) / sigma) ** 2)
    sensitivity /= np.max(sensitivity)
    sensitivity *= ch["max_val"]
    sensitivity_all.append(sensitivity)

# スタック（shape: (len(wl), 2)）
sensitivity_all = np.stack(sensitivity_all, axis=1)

# 合成：単純加算 → 正規化
combined_sensitivity = np.sum(sensitivity_all, axis=1)
combined_sensitivity /= np.max(combined_sensitivity)

# 描画
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, sensitivity_all[:, 0], label='Channel 1 (500nm peak)', linestyle='--')
plt.plot(wavelengths, sensitivity_all[:, 1], label='Channel 2 (250nm peak)', linestyle='--')
plt.plot(wavelengths, combined_sensitivity, label='Combined Sensitivity', color='black', linewidth=2)
plt.title('Combined Spectral Sensitivity')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Relative Sensitivity')
plt.grid(True)
plt.legend()
plt.xlim(wl_start, wl_end)
plt.ylim(0, 1.05)
plt.show()



