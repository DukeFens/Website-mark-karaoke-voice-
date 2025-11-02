import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import json
from tqdm import tqdm
import shutil

# ==== 1. INPUT ====
original_path = "beat_eval_system/input/original.wav"
record_path = "beat_eval_system/input/record.wav"

# ==== 1A. AUTO-CONVERT TO WAV ====
def ensure_wav_format(path):
    """Nếu file không phải .wav thì convert sang .wav và xóa bản gốc."""
    base, ext = os.path.splitext(path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        print(f"⚙️  Converting {path} → {wav_path}")
        y, sr = librosa.load(path, sr=None, mono=True)
        sf.write(wav_path, y, sr)
        try:
            os.remove(path)
            print(f"🗑️  Đã xóa file gốc: {path}")
        except Exception as e:
            print(f"⚠️ Không thể xóa file {path}: {e}")
        return wav_path
    return path

original_path = ensure_wav_format(original_path)
record_path = ensure_wav_format(record_path)

print("Đang đọc file:", original_path)
print("Đang đọc file:", record_path)

# ==== 2. PREPROCESSING ====
def preprocess_audio(path, target_sr=16000):
    # 1️⃣ Load audio
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2️⃣ Trim silence (remove leading/trailing silence)
    y, _ = librosa.effects.trim(y, top_db=30)

    # 3️⃣ Gentle noise reduction (adaptive)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.75, stationary=False)

    # 4️⃣ RMS-based loudness normalization
    rms = np.sqrt(np.mean(y**2))
    target_rms = 0.1  # ≈ -20 dBFS, studio level
    if rms > 0:
        y = y * (target_rms / rms)

    # 5️⃣ Safety limiter (avoid clipping)
    y = np.clip(y, -1.0, 1.0)

    return y, sr

y_ref, sr = preprocess_audio(original_path)
y_user, sr = preprocess_audio(record_path)

# ==== 3. FEATURE EXTRACTION ====
def extract_features(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

feat_ref = extract_features(y_ref, sr)
feat_user = extract_features(y_user, sr)

# ==== 4. SAVE PREPROCESSED FILES ====
os.makedirs("beat_eval_system/output", exist_ok=True)
orig_pre = "beat_eval_system/output/preprocessed_original.wav"
rec_pre = "beat_eval_system/output/preprocessed_record.wav"
sf.write(orig_pre, y_ref, sr)
sf.write(rec_pre, y_user, sr)
print("✅  Đã lưu 2 file preprocessed, sẵn sàng cho model...")

# ==== 5. MODEL: Beat Transformer ====
from beat_transformer_module import process_with_beat_transformer
beat_output = process_with_beat_transformer(orig_pre)
print("Beat Transformer output:", beat_output)

# ==== 6. MODEL: SwiftF0 ====
from swift_f0_module import process_with_swiftf0
f0_ref, f0_user = process_with_swiftf0(orig_pre, rec_pre)

# ==== 7. ALIGNMENT & METRIC ENGINE ====
from alignment_metric_module import run_alignment_metric

result_metrics = run_alignment_metric(
    f0_user, f0_ref,
    y_user=y_user,
    y_ref=y_ref,
    beat_times=beat_output['beat_times']
)

print("✅ Alignment & Metric Engine output:", result_metrics)

# ==== 8. SAVE FINAL RESULT ====
output_path = "beat_eval_system/output/result.json"

def np_convert(o):
    import numpy as np
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64, np.integer)):
        return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return str(o)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_metrics, f, indent=4, default=np_convert)

print(f"✅ Tất cả kết quả đã lưu vào {output_path}")
