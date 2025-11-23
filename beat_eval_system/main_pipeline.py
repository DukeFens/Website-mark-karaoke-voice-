"""
HỆ THỐNG ĐÁNH GIÁ GIỌNG HÁT – PIPELINE XỬ LÝ TỔNG THỂ
======================================================

Tổng quan:
-----------
Tệp `main_pipeline.py` đảm nhiệm vai trò là pipeline trung tâm của hệ thống đánh giá giọng hát.
Pipeline này tiếp nhận dữ liệu âm thanh đầu vào (file gốc và file người hát lại), tiến hành:
1. Chuẩn hóa định dạng tệp và tần số lấy mẫu.
2. Tiền xử lý (cắt khoảng lặng, khử nhiễu, chuẩn hóa âm lượng).
3. Chạy mô hình Beat Transformer (phát hiện nhịp và downbeat).
4. Chạy mô hình SwiftF0 (trích xuất cao độ).
5. Thực hiện căn chỉnh và tính toán các chỉ số đánh giá tổng hợp.
6. Xuất toàn bộ kết quả định lượng ra tệp JSON.
"""

import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import json
from tqdm import tqdm
import shutil
from similarity_score import compute_similarity

# ==== 1. XÁC ĐỊNH ĐƯỜNG DẪN INPUT ====
original_path = "beat_eval_system/input/original.wav"
record_path = "beat_eval_system/input/record.wav"


def ensure_wav_format(path):
    """
    Đảm bảo tệp âm thanh có định dạng WAV.  
    Nếu phát hiện tệp không phải .wav, hàm sẽ tự động chuyển đổi sang .wav và xóa tệp gốc.

    Thông số
    --------
    path : str  
        Đường dẫn đến tệp âm thanh cần kiểm tra hoặc chuyển đổi.

    Trả về
    -------
    str  
        Đường dẫn đến tệp .wav sau khi xử lý.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        print(f"🔄 Đang chuyển đổi {path} → {wav_path}")
        y, sr = librosa.load(path, sr=None, mono=True)
        sf.write(wav_path, y, sr)
        try:
            os.remove(path)
            print(f"🗑️  Đã xóa tệp gốc: {path}")
        except Exception as e:
            print(f"⚠️  Không thể xóa tệp {path}: {e}")
        return wav_path
    return path

# Chạy auto convert cho cả hai đầu vào
original_path = ensure_wav_format(original_path)
record_path = ensure_wav_format(record_path)

# ==== KIỂM TRA FILE RECORD CÓ BỊ RỖNG KHÔNG ====
if os.path.getsize(record_path) < 1000:  # <1 KB gần như chắc chắn là rỗng
    raise ValueError(
        f"❌ File record.wav bị rỗng hoặc hỏng: {record_path}\n"
        "Vui lòng kiểm tra ứng dụng thu âm hoặc đảm bảo file được ghi đúng."
    )

def preprocess_audio(path, target_sr=16000):
    """
    Tiền xử lý âm thanh: chuẩn hóa tần số, loại bỏ nhiễu, cân bằng âm lượng.

    Thông số
    --------
    path : str  
        Đường dẫn tới tệp âm thanh cần tiền xử lý.  
    target_sr : int, mặc định = 16000  
        Tần số lấy mẫu đích (Hz).

    Trả về
    -------
    y : np.ndarray  
        Mảng tín hiệu âm thanh sau khi xử lý.  
    sr : int  
        Tần số lấy mẫu sau khi chuẩn hóa.
    """
    # 1️⃣ Đọc dữ liệu âm thanh
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2️⃣ Cắt bỏ khoảng lặng ở đầu và cuối
    y, _ = librosa.effects.trim(y, top_db=30)

    # 3️⃣ Khử nhiễu nhẹ
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.75, stationary=False)

    # 4️⃣ Chuẩn hóa âm lượng
    rms = np.sqrt(np.mean(y**2))
    target_rms = 0.1
    if rms > 0:
        y = y * (target_rms / rms)

    # 5️⃣ Giới hạn biên độ
    y = np.clip(y, -1.0, 1.0)
    return y, sr

# Thực hiện tiền xử lý cho cả hai tệp âm thanh
y_ref, sr = preprocess_audio(original_path)
y_user, sr = preprocess_audio(record_path)

# ==== 4. LƯU TỆP ĐÃ TIỀN XỬ LÝ ====
os.makedirs("beat_eval_system/output", exist_ok=True)
orig_pre = "beat_eval_system/output/preprocessed_original.wav"
rec_pre = "beat_eval_system/output/preprocessed_record.wav"
sf.write(orig_pre, y_ref, sr)
sf.write(rec_pre, y_user, sr)


# ==== 5. CHẠY BEAT TRANSFORMER ====
from beat_transformer_module import process_with_beat_transformer
beat_output = process_with_beat_transformer(orig_pre)


# ==== 6. CHẠY SWIFT-F0 ====
from swift_f0_module import process_with_swiftf0
f0_ref, f0_user = process_with_swiftf0(orig_pre, rec_pre)


# ==== 7. CĂN CHỈNH & TÍNH CHỈ SỐ ====
from alignment_metric_module import run_alignment_metric

result_metrics = run_alignment_metric(
    f0_user,
    f0_ref,
    y_user=y_user,
    y_ref=y_ref,
    beat_times=beat_output["beat_times"],
)

# ==== TÍNH % SIMILARITY ====
similarity_percent = compute_similarity(result_metrics)
result_metrics["similarity_percent"] = similarity_percent

# ==== 8. LƯU JSON ====
output_path = "beat_eval_system/output/result.json"

def np_convert(o):
    """
    Chuyển đổi an toàn các kiểu dữ liệu NumPy sang định dạng JSON hợp lệ.
    """
    import numpy as np
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64, np.integer)):
        return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return str(o)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_metrics, f, indent=4, default=np_convert)
