"""
Pipeline chính của hệ thống chấm điểm hát (Singing Evaluation System).

Module này thực hiện toàn bộ quy trình đánh giá, bao gồm:

1. Kiểm tra đầu vào và chuẩn hóa định dạng WAV.
2. Tiền xử lý âm thanh: cắt lặng, resample, khử nhiễu, chuẩn hóa âm lượng.
3. Phân tích nhịp bằng Beat Transformer.
4. Trích xuất cao độ bằng SwiftF0.
5. Căn chỉnh và tính toán toàn bộ hệ thống metric.
6. Sinh feedback thông minh bằng LLM.
7. Xuất kết quả cuối cùng ra file JSON.

Toàn bộ hàm đều có docstring tiếng Việt theo chuẩn Google-style.
Không in ra console.
"""

import os
import json
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

from similarity_score import compute_similarity
from feedback_module import generate_feedback


# ---------------------------------------------------------------------------
# Đường dẫn đầu vào
# ---------------------------------------------------------------------------

ORIGINAL_PATH = "beat_eval_system/input/original.wav"
RECORD_PATH = "beat_eval_system/input/record.wav"


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

def ensure_wav_format(path):
    """Đảm bảo file âm thanh luôn ở định dạng WAV.

    Nếu file không phải WAV, hàm sẽ tải bằng Librosa và ghi lại thành WAV.
    File cũ được giữ nguyên.

    Args:
        path (str): Đường dẫn file âm thanh đầu vào.

    Returns:
        str: Đường dẫn file WAV mới hoặc file gốc nếu đã là WAV.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        audio, sr = librosa.load(path, sr=None, mono=True)
        sf.write(wav_path, audio, sr)
        return wav_path
    return path


def preprocess_audio(path, target_sr=16000):
    """Tiền xử lý âm thanh: resample, khử nhiễu, cắt lặng, chuẩn hóa.

    Các bước thực hiện:
    - Load mono.
    - Resample về 16kHz (mặc định).
    - Cắt vùng im lặng (silence trimming).
    - Khử nhiễu bằng noisereduce.
    - Chuẩn hóa RMS về mức ổn định.
    - Chặn clipping trong biên [-1, 1].

    Args:
        path (str): Đường dẫn file âm thanh.
        target_sr (int): Tần số lấy mẫu sau xử lý.

    Returns:
        tuple:
            processed_audio (np.ndarray): Âm thanh sau xử lý.
            sampling_rate (int): Tần số sau xử lý.
    """
    audio, sr = librosa.load(path, sr=None, mono=True)

    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Cắt lặng
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Khử nhiễu
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75, stationary=False)

    # Chuẩn hoá RMS
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 0.1
    if rms > 0:
        audio = audio * (target_rms / rms)

    # Chặn clipping
    audio = np.clip(audio, -1.0, 1.0)

    return audio, sr


def numpy_json_converter(obj):
    """Chuyển đổi an toàn kiểu NumPy sang kiểu Python để lưu JSON.

    Args:
        obj: Giá trị cần chuyển đổi.

    Returns:
        Any: Kiểu dữ liệu có thể tuần tự hoá JSON.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return str(obj)


# ---------------------------------------------------------------------------
# Chuẩn hóa đầu vào
# ---------------------------------------------------------------------------

ORIGINAL_PATH = ensure_wav_format(ORIGINAL_PATH)
RECORD_PATH = ensure_wav_format(RECORD_PATH)

if os.path.getsize(RECORD_PATH) < 1000:
    raise ValueError("record.wav quá ngắn hoặc bị hỏng.")


# ---------------------------------------------------------------------------
# Tiền xử lý âm thanh
# ---------------------------------------------------------------------------

y_ref, sr = preprocess_audio(ORIGINAL_PATH)
y_user, sr = preprocess_audio(RECORD_PATH)

os.makedirs("beat_eval_system/output", exist_ok=True)
ORIG_PRE = "beat_eval_system/output/preprocessed_original.wav"
REC_PRE = "beat_eval_system/output/preprocessed_record.wav"

sf.write(ORIG_PRE, y_ref, sr)
sf.write(REC_PRE, y_user, sr)


# ---------------------------------------------------------------------------
# Beat Transformer
# ---------------------------------------------------------------------------

from beat_transformer_module import process_with_beat_transformer
beat_output = process_with_beat_transformer(ORIG_PRE)


# ---------------------------------------------------------------------------
# SwiftF0 Pitch Extraction
# ---------------------------------------------------------------------------

from swift_f0_module import process_with_swiftf0
f0_ref, f0_user = process_with_swiftf0(ORIG_PRE, REC_PRE)


# ---------------------------------------------------------------------------
# Alignment + Metrics
# ---------------------------------------------------------------------------

from alignment_metric_module import run_alignment_metric

result_metrics = run_alignment_metric(
    f0_user=f0_user,
    f0_ref=f0_ref,
    y_user=y_user,
    y_ref=y_ref,
    beat_times=beat_output["beat_times"],
)

result_metrics["similarity_percent"] = compute_similarity(result_metrics)


# ---------------------------------------------------------------------------
# Sinh feedback (format dict)
# ---------------------------------------------------------------------------

try:
    feedback_dict = generate_feedback(result_metrics)
except Exception:
    feedback_dict = {"pitch": "", "timing": "", "energy": ""}

result_metrics["feedback"] = feedback_dict


# ---------------------------------------------------------------------------
# Xuất JSON
# ---------------------------------------------------------------------------

OUTPUT_PATH = "beat_eval_system/output/result.json"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result_metrics, f, indent=4, ensure_ascii=False, default=numpy_json_converter)
