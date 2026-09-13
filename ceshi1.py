import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import butter, filtfilt, find_peaks


# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "1.csv")
print(f"🔍 实际读取路径: {CSV_FILE_PATH}")

SAMPLE_RATE_HZ = 20000
SCAN_SPEED_MM_S = 4
VCC_V = 4.0
V0_MV = 0.0

F_CARRIER_HZ = 1000.0
ENABLE_DEMOD = True
ENVELOPE_CUTOFF_HZ = 30.0
FILTER_ORDER = 4

SENSITIVITY_MV_V_GS = 55.0
_SENSITIVITY_MV_PER_MT = SENSITIVITY_MV_V_GS * 10.0 * VCC_V
SENSITIVITY_MV_PER_UT = _SENSITIVITY_MV_PER_MT / 1000.0

ENABLE_BANDPASS = True
BP_LOW_HZ = 800.0
BP_HIGH_HZ = 1200.0

ENABLE_BASELINE_REMOVAL = True
BASELINE_WINDOW_MM = 30.0

ENABLE_SMOOTHING = True
SMOOTH_WINDOW_MM = 0.8

SIGNAL_GAIN = 5.0

ENABLE_PEAK_DETECT = True
PEAK_PROMINENCE_UT = 0.05
PEAK_MIN_DIST_MM = 10.0

DEFECT_LABEL_THRESHOLD_UT = -20.0


# ================= 执行区 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_csv(CSV_FILE_PATH, header=None, encoding='gbk',
                     usecols=[0], on_bad_lines='warn')
    voltage_mv = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
except Exception as e:
    print(f"❌ 读取文件失败: {e}")
    exit()

v_max = np.nanmax(np.abs(voltage_mv))
if v_max < 1.0:
    voltage_mv = voltage_mv * 1000.0
elif v_max > 10000.0:
    voltage_mv = voltage_mv / 1000.0

print(f"✅ 读取成功，共 {len(voltage_mv)} 个数据点")

time_s = np.arange(len(voltage_mv)) / SAMPLE_RATE_HZ
distance_mm = time_s * SCAN_SPEED_MM_S


# ================= 🔧 优化后的滤波函数库 =================
def lowpass_filter(data, cutoff_hz, fs, order=4):
    """IIR低通（本身已高效，无需改动）"""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, data)


def bandpass_filter(data, low, high, fs, order=4):
    """IIR带通（本身已高效，无需改动）"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)


def remove_baseline(data, window_points):
    """
    🔧 优化：用累积和代替 np.convolve
    原复杂度 O(N×W) → 现复杂度 O(N)
    边界处理与原 mode='same' 行为一致
    """
    if window_points < 3:
        return data, np.zeros_like(data)
    if window_points % 2 == 0:
        window_points += 1

    n = len(data)
    half = window_points // 2

    # 累积和计算滑动均值，O(N)
    cumsum = np.cumsum(data)
    baseline = np.empty(n, dtype=np.float64)

    # 左边界：窗口不足时用可用点数平均
    for i in range(min(half, n)):
        right = min(i + half, n - 1)
        count = right + 1
        baseline[i] = cumsum[right] / count

    # 中间区域：完整窗口
    start_full = max(half, 0)
    end_full = min(n - half, n)
    if end_full > start_full:
        left_cum = np.empty(end_full - start_full, dtype=np.float64)
        left_cum[0] = cumsum[start_full - half - 1] if (start_full - half - 1) >= 0 else 0.0
        left_cum[1:] = cumsum[start_full - half:end_full - half - 1]
        right_cum = cumsum[start_full + half:end_full + half]
        # 修正右边界截断
        valid_right = np.minimum(np.arange(start_full, end_full) + half, n - 1)
        right_cum = cumsum[valid_right]
        baseline[start_full:end_full] = (right_cum - left_cum) / window_points

    # 右边界
    for i in range(max(n - half, 0), n):
        left = max(i - half, 0)
        count = min(i + half, n - 1) - left + 1
        s = cumsum[min(i + half, n - 1)] - (cumsum[left - 1] if left > 0 else 0.0)
        baseline[i] = s / count

    return data - baseline, baseline


def smooth_signal(data, window_points):
    """
    🔧 优化：用累积和代替 np.convolve
    原复杂度 O(N×W) → 现复杂度 O(N)
    """
    if window_points < 3:
        return data
    if window_points % 2 == 0:
        window_points += 1

    n = len(data)
    half = window_points // 2
    cumsum = np.cumsum(data)
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        left = max(i - half, 0)
        right = min(i + half, n - 1)
        count = right - left + 1
        s = cumsum[right] - (cumsum[left - 1] if left > 0 else 0.0)
        result[i] = s / count

    return result


# ================= 数字解调 =================
def demodulate(signal, fs, f_carrier, cutoff_hz, order=4):
    t = np.arange(len(signal)) / fs
    ref_sin = np.sin(2 * np.pi * f_carrier * t)
    ref_cos = np.cos(2 * np.pi * f_carrier * t)
    mixed_i = signal * ref_sin
    mixed_q = signal * ref_cos
    env_i = lowpass_filter(mixed_i, cutoff_hz, fs, order)
    env_q = lowpass_filter(mixed_q, cutoff_hz, fs, order)
    envelope = np.sqrt(env_i ** 2 + env_q ** 2)
    return envelope, env_i, env_q


# ================= 信号增强主流程 =================
if ENABLE_BANDPASS:
    voltage_clean = bandpass_filter(voltage_mv, BP_LOW_HZ, BP_HIGH_HZ, SAMPLE_RATE_HZ, FILTER_ORDER)
else:
    voltage_clean = voltage_mv

if ENABLE_DEMOD:
    envelope_mv, _, _ = demodulate(voltage_clean, SAMPLE_RATE_HZ, F_CARRIER_HZ,
                                   ENVELOPE_CUTOFF_HZ, FILTER_ORDER)
    B_ut_raw = (envelope_mv - V0_MV) / SENSITIVITY_MV_PER_UT
else:
    B_ut_raw = (voltage_clean - V0_MV) / SENSITIVITY_MV_PER_UT

B_ut_enhanced = B_ut_raw.copy()
if ENABLE_BASELINE_REMOVAL:
    win_pts = int(BASELINE_WINDOW_MM / SCAN_SPEED_MM_S * SAMPLE_RATE_HZ)
    B_ut_enhanced, baseline = remove_baseline(B_ut_enhanced, win_pts)

if ENABLE_SMOOTHING:
    sm_pts = int(SMOOTH_WINDOW_MM / SCAN_SPEED_MM_S * SAMPLE_RATE_HZ)
    B_ut_enhanced = smooth_signal(B_ut_enhanced, sm_pts)

B_ut_enhanced = B_ut_enhanced * SIGNAL_GAIN


# ================= 自动波谷检测 =================
peak_positions_mm = []
peak_values_ut = []
if ENABLE_PEAK_DETECT:
    min_dist_pts = int(PEAK_MIN_DIST_MM / SCAN_SPEED_MM_S * SAMPLE_RATE_HZ)
    peaks_idx, _ = find_peaks(-B_ut_enhanced,
                              prominence=PEAK_PROMINENCE_UT * SIGNAL_GAIN,
                              distance=max(min_dist_pts, 1))
    peak_positions_mm = distance_mm[peaks_idx]
    peak_values_ut = B_ut_enhanced[peaks_idx]

    valid_defects = [(p, v) for p, v in zip(peak_positions_mm, peak_values_ut)
                     if v < DEFECT_LABEL_THRESHOLD_UT]
    print(f"🔍 检测到 {len(valid_defects)} 个疑似裂纹波谷（y < {DEFECT_LABEL_THRESHOLD_UT} μT）:")
    for i, (pos, val) in enumerate(valid_defects):
        print(f"   缺陷{i+1}: 位置={pos:.1f} mm, 幅值={val:.4f} μT")


# ================= 绘图（两子图）=================
plt.close('all')
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# ---------- 子图1: 解调包络 + 基线 ----------
axes[0].plot(distance_mm, B_ut_raw, color='#ff7f0e', linewidth=0.8, label='解调包络（未增强）')
if ENABLE_BASELINE_REMOVAL:
    axes[0].plot(distance_mm, baseline, color='k', linewidth=1.2,
                 linestyle='-.', alpha=0.7, label='估计基线')

axes[0].set_title('① 锁相解调包络 + 估计基线', fontsize=16, fontweight='bold', pad=12)
axes[0].set_ylabel('磁场梯度 (μT)', fontsize=14, fontweight='bold', labelpad=10)
axes[0].tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(fontsize=13, loc='best', framealpha=0.9)

# ---------- 子图2: 增强后信号 ----------
axes[1].plot(distance_mm, B_ut_enhanced, color='#d62728', linewidth=1.0,
             label=f'增强后信号 (×{SIGNAL_GAIN})')
axes[1].axhline(y=0, color='k', linewidth=0.6, linestyle='-', alpha=0.4)
axes[1].axhline(y=DEFECT_LABEL_THRESHOLD_UT, color='darkred', linewidth=0.8,
                linestyle='-.', alpha=0.6, label=f'标注阈值 ({DEFECT_LABEL_THRESHOLD_UT} μT)')

# --- 缺陷标注 ---
valid_label_idx = 0
y_tick_positions = []
y_tick_labels = []
for pos, val in zip(peak_positions_mm, peak_values_ut):
    if val >= DEFECT_LABEL_THRESHOLD_UT:
        continue
    valid_label_idx += 1
    y_tick_positions.append(val)
    y_tick_labels.append(f'缺陷{valid_label_idx}\n{val:.4f}μT')
    axes[1].axhline(y=val, color='darkred', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[1].plot(pos, val, 'o', color='darkred', markersize=5, zorder=5)

# --- Y轴刻度合并与样式 ---
existing_ticks = axes[1].get_yticks()
merged_ticks = np.unique(np.concatenate([existing_ticks, y_tick_positions]))
axes[1].set_yticks(merged_ticks)

tick_label_map = {v: f'{v:.4f}' for v in existing_ticks}
for y_val, label in zip(y_tick_positions, y_tick_labels):
    tick_label_map[y_val] = label

final_labels = [tick_label_map.get(t, f'{t:.4f}') for t in merged_ticks]
axes[1].set_yticklabels(final_labels, fontsize=12)

# 高亮缺陷刻度
for tick_label, y_val in zip(axes[1].get_yticklabels(), merged_ticks):
    if y_val in y_tick_positions:
        tick_label.set_color('darkred')
        tick_label.set_fontweight('bold')
        tick_label.set_fontsize(13)       # 🔧 缺陷标签额外加大
    else:
        tick_label.set_color('black')
        tick_label.set_fontsize(12)

axes[1].set_title('② 增强后 MRPT 信号（裂纹波谷清晰可见）', fontsize=16, fontweight='bold', pad=12)
axes[1].set_xlabel('扫描位移 (mm)', fontsize=14, fontweight='bold', labelpad=10)
axes[1].set_ylabel('磁场梯度 (μT)', fontsize=14, fontweight='bold', labelpad=10)
axes[1].tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(fontsize=13, loc='best', framealpha=0.9)
axes[1].autoscale(enable=True, axis='y')

plt.tight_layout(h_pad=3.0)   # 🔧 增大子图间距，避免文字重叠
plt.show()

print(f"\n📏 总扫描距离: {distance_mm[-1]:.2f} mm")
print(f"📈 增强后范围: {B_ut_enhanced.min():.3f} ~ {B_ut_enhanced.max():.3f} μT")
print("\n✅ 处理完成。")
sys.exit(0)