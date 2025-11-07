import torch
import librosa
import numpy as np
from madmom.features import DBNBeatTrackingProcessor, RNNBeatProcessor

# ============================================================
#  MODULE: Beat Transformer
#  Chức năng: Xác định nhịp (beat), nhịp mạnh (downbeat) và
#  ước lượng tempo (BPM) của đoạn âm thanh sau khi tiền xử lý.
# ============================================================

def process_with_beat_transformer(audio_path):
    """
    Hàm chính của Beat Transformer.

    Đầu vào:
        audio_path (str): đường dẫn tới tệp âm thanh đã được tiền xử lý 
                          (thường là 'preprocessed_original.wav').

    Đầu ra:
        dict chứa các trường:
            - "beat_times": danh sách thời điểm xuất hiện từng nhịp (tính bằng giây)
            - "downbeat_flags": mảng đánh dấu nhịp mạnh (1 là nhịp mạnh, 0 là nhịp thường)
            - "tempo": giá trị tempo trung bình (BPM)
    """

    # --- 1. Tải âm thanh ---
    #   Giữ nguyên tần số mẫu gốc (sr=None), chuyển sang mono để đơn giản hóa xử lý.
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # --- 2. Mô hình suy luận (RNN/Transformer-based beat tracking) ---
    #   Sử dụng mô hình RNN được huấn luyện sẵn trong thư viện madmom.
    proc = RNNBeatProcessor()
    beat_act = proc(str(audio_path))

    # --- 3. Giải mã theo thời gian (Temporal decoding bằng DBN) ---
    #   DBN giúp tinh chỉnh kết quả đầu ra, đảm bảo tính ổn định về thời gian.
    tracker = DBNBeatTrackingProcessor(fps=100)
    beat_times = tracker(beat_act)

    # --- 4. Ước lượng tempo ---
    #   Tính trung vị khoảng cách giữa các nhịp → suy ra tempo (BPM).
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        tempo = 60.0 / np.median(intervals)
    else:
        tempo = 0

    # --- 5. Xác định nhịp mạnh (downbeat) ---
    #   Giả định nhịp 4/4 → mỗi 4 nhịp thì nhịp đầu là downbeat.
    downbeat_flags = np.zeros(len(beat_times))
    downbeat_flags[::4] = 1

    # --- 6. Trả kết quả ---
    return {
        "beat_times": beat_times.tolist(),
        "downbeat_flags": downbeat_flags.tolist(),
        "tempo": float(tempo)
    }


# ============================================================
#  KIỂM THỬ TRỰC TIẾP MODULE
# ============================================================
if __name__ == "__main__":
    output = process_with_beat_transformer("beat_eval_system/output/preprocessed_original.wav")
    print("Kết quả Beat Transformer:")
    print(output)
