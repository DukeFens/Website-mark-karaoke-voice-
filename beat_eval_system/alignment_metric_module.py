# ============================================================
#  MODULE: Alignment & Metric Computation
#  Chức năng:
#     - Căn chỉnh cao độ (F0) giữa mẫu gốc và bản thu bằng DTW.
#     - Tính toán các chỉ số đánh giá (pitch, timing, năng lượng, nhịp, âm sắc).
# ============================================================

import numpy as np
from fastdtw import fastdtw
import librosa

# ============================================================
#  HÀM PHỤ TRỢ CHUYỂN ĐỔI VÀ XỬ LÝ DỮ LIỆU
# ============================================================

def _hz_to_cents_safe(f_user_hz, f_ref_hz):
    """
    Chuyển đổi tần số (Hz) sang đơn vị cent (so sánh cao độ), 
    đồng thời xử lý an toàn với giá trị 0 hoặc NaN.

    Đầu vào:
        f_user_hz (array-like): mảng tần số của người dùng
        f_ref_hz (array-like): mảng tần số tham chiếu (mẫu gốc)

    Đầu ra:
        np.ndarray: sai lệch cao độ (cent) giữa hai dãy
    """
    fu = np.asarray(f_user_hz, dtype=np.float64)
    fr = np.asarray(f_ref_hz, dtype=np.float64)

    # Thay thế giá trị không hợp lệ để tránh chia cho 0
    fu_mask = np.isnan(fu) | (fu <= 0)
    fr_mask = np.isnan(fr) | (fr <= 0)
    invalid = fr_mask | fu_mask
    fu_safe = np.where(invalid, np.nan, fu)
    fr_safe = np.where(invalid, np.nan, fr)

    cents = 1200.0 * np.log2(fu_safe / fr_safe)
    return cents  # giữ nguyên NaN ở vị trí không hợp lệ


def _safe_mean(x):
    """Tính trung bình bỏ qua NaN."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    return float(np.mean(x))


def _safe_std(x):
    """Tính độ lệch chuẩn bỏ qua NaN."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    return float(np.std(x))


# ============================================================
#  CĂN KHUNG (FRAME MAPPING)
# ============================================================

def frame_mapping(user_f0, ref_f0):
    """
    Ánh xạ độ dài chuỗi F0 của người dùng sang cùng số khung với mẫu gốc 
    bằng nội suy tuyến tính (linear interpolation).

    Đầu ra:
        np.ndarray: mảng F0 của người dùng đã được ánh xạ lại
    """
    user = np.asarray(user_f0, dtype=np.float64).flatten()
    ref = np.asarray(ref_f0, dtype=np.float64).flatten()

    user_valid = np.nan_to_num(user, nan=0.0)
    if len(user_valid) == 0 or len(ref) == 0:
        return np.array([], dtype=np.float64)

    if len(user_valid) != len(ref):
        x_user = np.linspace(0.0, 1.0, len(user_valid))
        x_ref = np.linspace(0.0, 1.0, len(ref))
        mapped = np.interp(x_ref, x_user, user_valid)
    else:
        mapped = user_valid.copy()

    return mapped


# ============================================================
#  CĂN CHỈNH TRÌNH TỰ (DTW ALIGNMENT)
# ============================================================

def weighted_alignment(user_f0, ref_f0):
    """
    Thực hiện Dynamic Time Warping (DTW) trên hai chuỗi F0 (1 chiều).

    Đầu ra:
        tuple gồm:
            - user_aligned (np.ndarray): F0 người dùng sau căn chỉnh
            - ref_aligned (np.ndarray): F0 mẫu gốc tương ứng
            - distance (float): khoảng cách DTW
    """
    user = np.nan_to_num(np.asarray(user_f0, dtype=np.float64).flatten(), nan=0.0)
    ref = np.nan_to_num(np.asarray(ref_f0, dtype=np.float64).flatten(), nan=0.0)

    if user.size == 0 or ref.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.0

    distance, path = fastdtw(user, ref, dist=lambda a, b: abs(a - b))
    user_aligned = np.array([user[i] for i, _ in path], dtype=np.float64)
    ref_aligned = np.array([ref[j] for _, j in path], dtype=np.float64)

    return user_aligned, ref_aligned, float(distance)


# ============================================================
#  HÀM TRÍCH XUẤT ĐẶC TRƯNG (FEATURE HELPERS)
# ============================================================

def compute_energy(y, hop_length=256, frame_length=1024):
    """Tính năng lượng RMS theo khung (trả về mảng 1D)."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)
    return rms.flatten()


def compute_mfcc_mean(y, sr, hop_length=256, n_mfcc=13):
    """Tính giá trị trung bình của MFCC (đặc trưng âm sắc) theo thời gian."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.size == 0:
        return None
    return np.mean(mfcc, axis=1)


def extract_onsets_from_wave(y, sr, hop_length=256):
    """Trích xuất thời điểm bắt đầu nốt (onset) từ waveform bằng librosa."""
    return librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='time')


# ============================================================
#  TÍNH TOÁN CÁC CHỈ SỐ (METRIC COMPUTATION)
# ============================================================

def compute_metrics(user_f0_aligned, ref_f0_aligned,
                    y_user=None, y_ref=None,
                    sr=16000, hop_length=160,
                    beat_times=None,
                    user_onsets=None, ref_onsets=None):
    """
    Tính toán các chỉ số đánh giá (metrics) dựa trên dữ liệu đã căn chỉnh.

    Trả về:
        dict chứa các trường:
            - pitch_deviation_mean, pitch_deviation_std (cent)
            - timing_deviation (ms)
            - onbeat_accuracy (%)
            - energy_deviation_mean, energy_deviation_std
            - formant_stability
            - rhythmic_variation
            - dtw_distance (thêm bởi caller)
    """
    metrics = {}

    # --- 1. Sai lệch cao độ ---
    user = np.asarray(user_f0_aligned, dtype=np.float64)
    ref = np.asarray(ref_f0_aligned, dtype=np.float64)
    if user.size == 0 or ref.size == 0:
        metrics['pitch_deviation_mean'] = None
        metrics['pitch_deviation_std'] = None
    else:
        cents = _hz_to_cents_safe(user, ref)
        metrics['pitch_deviation_mean'] = _safe_mean(np.abs(cents))
        metrics['pitch_deviation_std'] = _safe_std(cents)

    # --- 2. Trích xuất onsets nếu chưa có ---
    if user_onsets is None and y_user is not None:
        try:
            user_onsets = extract_onsets_from_wave(y_user, sr=sr, hop_length=hop_length)
        except Exception:
            user_onsets = np.array([])
    if ref_onsets is None and y_ref is not None:
        try:
            ref_onsets = extract_onsets_from_wave(y_ref, sr=sr, hop_length=hop_length)
        except Exception:
            ref_onsets = np.array([])

    # --- 3. Sai lệch thời gian & độ chính xác theo nhịp ---
    if beat_times is not None and user_onsets is not None and len(user_onsets) > 0:
        beat_times_arr = np.asarray(beat_times, dtype=np.float64)
        user_onsets_arr = np.asarray(user_onsets, dtype=np.float64)

        diffs = []
        for o in user_onsets_arr:
            idx = np.argmin(np.abs(beat_times_arr - o))
            diffs.append(abs(o - beat_times_arr[idx]))
        diffs = np.array(diffs, dtype=np.float64)

        metrics['timing_deviation'] = float(np.mean(diffs) * 1000.0) if diffs.size > 0 else None  # đơn vị: ms
        onbeat_thresh = 0.05  # dung sai 50 ms
        metrics['onbeat_accuracy'] = float(100.0 * np.sum(diffs <= onbeat_thresh) / diffs.size) if diffs.size > 0 else None
    else:
        metrics['timing_deviation'] = None
        metrics['onbeat_accuracy'] = None

    # --- 4. Sai lệch năng lượng ---
    if (y_user is not None) and (y_ref is not None):
        ue = compute_energy(y_user, hop_length=hop_length)
        re = compute_energy(y_ref, hop_length=hop_length)
        if ue.size == 0 or re.size == 0:
            metrics['energy_deviation_mean'] = None
            metrics['energy_deviation_std'] = None
        else:
            min_len = min(len(ue), len(re))
            diff = np.abs(ue[:min_len] - re[:min_len])
            metrics['energy_deviation_mean'] = float(np.mean(diff))
            metrics['energy_deviation_std'] = float(np.std(diff))
    else:
        metrics['energy_deviation_mean'] = None
        metrics['energy_deviation_std'] = None

    # --- 5. Ổn định âm sắc (MFCC trung bình) ---
    if (y_user is not None) and (y_ref is not None):
        mfcc_user_mean = compute_mfcc_mean(y_user, sr=sr, hop_length=hop_length)
        mfcc_ref_mean = compute_mfcc_mean(y_ref, sr=sr, hop_length=hop_length)
        if mfcc_user_mean is None or mfcc_ref_mean is None:
            metrics['formant_stability'] = None
        else:
            minlen = min(len(mfcc_user_mean), len(mfcc_ref_mean))
            metrics['formant_stability'] = float(np.mean(np.abs(mfcc_user_mean[:minlen] - mfcc_ref_mean[:minlen])))
    else:
        metrics['formant_stability'] = None

    # --- 6. Dao động nhịp (rhythmic variation) ---
    if user_onsets is not None and len(user_onsets) > 1:
        ioi = np.diff(user_onsets)
        metrics['rhythmic_variation'] = float(np.std(ioi))
    else:
        metrics['rhythmic_variation'] = None

    return metrics


# ============================================================
#  GÓI TỔNG HỢP: ALIGNMENT + METRIC
# ============================================================

def run_alignment_metric(f0_user, f0_ref, y_user=None, y_ref=None, beat_times=None, sr=16000, hop_length=160):
    """
    Thực hiện toàn bộ quy trình:
        1) Căn khung F0 (frame mapping)
        2) Căn chỉnh F0 bằng DTW
        3) Tính toán các chỉ số đánh giá

    Đầu ra:
        dict chứa các chỉ số (float hoặc None)
    """
    # 1) Ánh xạ độ dài
    f0_user_mapped = frame_mapping(f0_user, f0_ref)

    # 2) Căn chỉnh bằng DTW
    f0_user_aligned, f0_ref_aligned, distance = weighted_alignment(f0_user_mapped, f0_ref)

    # 3) Tính toán chỉ số
    metrics = compute_metrics(f0_user_aligned, f0_ref_aligned,
                              y_user=y_user, y_ref=y_ref,
                              sr=sr, hop_length=hop_length,
                              beat_times=beat_times)

    # 4) Thêm giá trị khoảng cách DTW
    metrics['dtw_distance'] = float(distance)

    return metrics
