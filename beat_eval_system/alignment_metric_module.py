# beat_eval_system/alignment_metric_module.py
import numpy as np
from fastdtw import fastdtw
import librosa

# --------- Helpers ----------
def _hz_to_cents_safe(f_user_hz, f_ref_hz):
    """Convert arrays of Hz to cents (elementwise), handle zeros/NaN safely."""
    fu = np.asarray(f_user_hz, dtype=np.float64)
    fr = np.asarray(f_ref_hz, dtype=np.float64)

    # Replace non-positive or NaN with small positive to avoid divide-by-zero
    fu_mask = np.isnan(fu) | (fu <= 0)
    fr_mask = np.isnan(fr) | (fr <= 0)

    # Where reference is invalid, we can't compute cents reliably -> mark NaN
    invalid = fr_mask | fu_mask
    fu_safe = np.where(invalid, np.nan, fu)
    fr_safe = np.where(invalid, np.nan, fr)

    cents = 1200.0 * np.log2(fu_safe / fr_safe)
    # keep NaNs where invalid
    return cents

def _safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    return float(np.mean(x))

def _safe_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    return float(np.std(x))

# --------- Frame mapping ----------
def frame_mapping(user_f0, ref_f0):
    """
    Map user_f0 to the same frame length as ref_f0 via linear interpolation.
    Returns mapped_user_f0 (numpy array).
    """
    user = np.asarray(user_f0, dtype=np.float64).flatten()
    ref = np.asarray(ref_f0, dtype=np.float64).flatten()

    # Keep NaNs in place for interpolation (interp requires non-NaN; fill temporarily)
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

# --------- DTW alignment ----------
def weighted_alignment(user_f0, ref_f0):
    """
    DTW on 1-D sequences. Returns aligned user/ref arrays and distance (float).
    """
    user = np.nan_to_num(np.asarray(user_f0, dtype=np.float64).flatten(), nan=0.0)
    ref = np.nan_to_num(np.asarray(ref_f0, dtype=np.float64).flatten(), nan=0.0)

    if user.size == 0 or ref.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.0

    distance, path = fastdtw(user, ref, dist=lambda a, b: abs(a - b))

    user_aligned = np.array([user[i] for i, _ in path], dtype=np.float64)
    ref_aligned = np.array([ref[j] for _, j in path], dtype=np.float64)
    return user_aligned, ref_aligned, float(distance)

# --------- Feature helpers ----------
def compute_energy(y, hop_length=256, frame_length=1024):
    """Return RMS per frame (1D numpy)."""
    # librosa.feature.rms returns shape (1, n_frames)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)
    return rms.flatten()

def compute_mfcc_mean(y, sr, hop_length=256, n_mfcc=13):
    """Return MFCC mean vector over time (1D length n_mfcc)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.size == 0:
        return None
    return np.mean(mfcc, axis=1)

def extract_onsets_from_wave(y, sr, hop_length=256):
    """Return onset times (seconds) from waveform using librosa onset detection."""
    # onset detection works on waveform; units='time' returns seconds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='time')
    return onsets

# --------- Metric computation ----------
def compute_metrics(user_f0_aligned, ref_f0_aligned,
                    y_user=None, y_ref=None,
                    sr=16000, hop_length=160,
                    beat_times=None,
                    user_onsets=None, ref_onsets=None):
    """
    Compute metrics and return dict of values (native Python floats / lists / None).
    Fields returned:
      pitch_deviation_mean, pitch_deviation_std (cents)
      timing_deviation (ms), onbeat_accuracy (%)
      energy_deviation_mean, energy_deviation_std
      formant_stability, rhythmic_variation, dtw_distance (set by caller)
    """
    metrics = {}

    # Pitch deviation in cents (elementwise)
    user = np.asarray(user_f0_aligned, dtype=np.float64)
    ref = np.asarray(ref_f0_aligned, dtype=np.float64)
    if user.size == 0 or ref.size == 0:
        metrics['pitch_deviation_mean'] = None
        metrics['pitch_deviation_std'] = None
    else:
        cents = _hz_to_cents_safe(user, ref)
        metrics['pitch_deviation_mean'] = _safe_mean(np.abs(cents))
        metrics['pitch_deviation_std'] = _safe_std(cents)

    # Onsets: prefer provided; else extract from waveform
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

    # Timing deviation & On-beat accuracy (requires beat_times + user_onsets)
    if beat_times is not None and user_onsets is not None and len(user_onsets) > 0:
        beat_times_arr = np.asarray(beat_times, dtype=np.float64)
        user_onsets_arr = np.asarray(user_onsets, dtype=np.float64)
        # For each onset find nearest beat
        diffs = []
        for o in user_onsets_arr:
            idx = np.argmin(np.abs(beat_times_arr - o))
            diffs.append(abs(o - beat_times_arr[idx]))
        diffs = np.array(diffs, dtype=np.float64)
        metrics['timing_deviation'] = float(np.mean(diffs) * 1000.0) if diffs.size > 0 else None  # ms
        onbeat_thresh = 0.05  # 50 ms tolerance
        metrics['onbeat_accuracy'] = float(100.0 * np.sum(diffs <= onbeat_thresh) / diffs.size) if diffs.size > 0 else None
    else:
        metrics['timing_deviation'] = None
        metrics['onbeat_accuracy'] = None

    # Energy deviation - compute RMS frames and align by min length
    if (y_user is not None) and (y_ref is not None):
        ue = compute_energy(y_user, hop_length=hop_length)
        re = compute_energy(y_ref, hop_length=hop_length)
        if ue.size == 0 or re.size == 0:
            metrics['energy_deviation_mean'] = None
            metrics['energy_deviation_std'] = None
        else:
            min_len = min(len(ue), len(re))
            ue = ue[:min_len]
            re = re[:min_len]
            diff = np.abs(ue - re)
            metrics['energy_deviation_mean'] = float(np.mean(diff))
            metrics['energy_deviation_std'] = float(np.std(diff))
    else:
        metrics['energy_deviation_mean'] = None
        metrics['energy_deviation_std'] = None

    # Formant / timbre stability - use MFCC mean distance (robust)
    if (y_user is not None) and (y_ref is not None):
        mfcc_user_mean = compute_mfcc_mean(y_user, sr=sr, hop_length=hop_length)
        mfcc_ref_mean = compute_mfcc_mean(y_ref, sr=sr, hop_length=hop_length)
        if mfcc_user_mean is None or mfcc_ref_mean is None:
            metrics['formant_stability'] = None
        else:
            # if different lengths, pad/truncate to min
            minlen = min(len(mfcc_user_mean), len(mfcc_ref_mean))
            metrics['formant_stability'] = float(np.mean(np.abs(mfcc_user_mean[:minlen] - mfcc_ref_mean[:minlen])))
    else:
        metrics['formant_stability'] = None

    # Rhythmic variation: std of IOI (inter-onset intervals)
    if user_onsets is not None and len(user_onsets) > 1:
        ioi = np.diff(user_onsets)
        metrics['rhythmic_variation'] = float(np.std(ioi))
    else:
        metrics['rhythmic_variation'] = None

    return metrics

# --------- Run alignment + metrics ----------
def run_alignment_metric(f0_user, f0_ref, y_user=None, y_ref=None, beat_times=None, sr=16000, hop_length=160):
    """
    High-level wrapper: map frames, align F0 via DTW, compute metrics.
    Returns a dict of metrics with native Python types (float or None).
    """
    # 1) Frame mapping: match user f0 length to ref f0 length
    f0_user_mapped = frame_mapping(f0_user, f0_ref)

    # 2) DTW alignment (weighted)
    f0_user_aligned, f0_ref_aligned, distance = weighted_alignment(f0_user_mapped, f0_ref)

    # 3) Compute metrics (pass waveform so energy/formant/onsets can be derived)
    metrics = compute_metrics(f0_user_aligned, f0_ref_aligned,
                              y_user=y_user, y_ref=y_ref,
                              sr=sr, hop_length=hop_length,
                              beat_times=beat_times)

    # 4) attach DTW distance (float)
    metrics['dtw_distance'] = float(distance)

    # Convert None -> null semantics; keep Python types (json.dump handles None->null)
    return metrics
