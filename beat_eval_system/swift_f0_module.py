import torch
import numpy as np
import os
import soundfile as sf
import torchaudio

# ============================================================
#  MODULE: SwiftF0
#  Chức năng: Trích xuất cao độ (pitch/F0) nhanh và chính xác 
#  từ các đoạn âm thanh giọng hát, phục vụ so sánh giữa mẫu gốc 
#  (original) và bản thu của người dùng (record).
#
#  Tham khảo:
#  https://github.com/w-okada/voice-changer/blob/main/Realtime-Voice-Clone/swift_pitch_extraction.py
# ============================================================


def load_audio_sf(audio_path):
    """
    Đọc tệp WAV bằng thư viện `soundfile` để tránh lỗi codec của Torch trên Windows.

    Đầu vào:
        audio_path (str): đường dẫn tới tệp âm thanh (.wav)

    Đầu ra:
        tuple gồm:
            - waveform (torch.Tensor): dạng sóng âm [1, num_samples]
            - sr (int): tần số lấy mẫu (sample rate)
    """
    y, sr = sf.read(audio_path)
    if y.ndim > 1:  # Nếu là stereo → chuyển sang mono
        y = np.mean(y, axis=1)
    waveform = torch.from_numpy(y).unsqueeze(0).float()
    return waveform, sr


def extract_pitch_swift(audio_path, hop_length=160, fmin=50.0, fmax=1100.0):
    """
    Trích xuất cao độ (F0) từ tệp âm thanh bằng thuật toán SwiftF0 trong torchaudio.

    Đầu vào:
        audio_path (str): đường dẫn tới tệp .wav
        hop_length (int): độ dài khung nhảy (số mẫu giữa 2 khung)
        fmin (float): tần số thấp nhất có thể phát hiện (Hz)
        fmax (float): tần số cao nhất có thể phát hiện (Hz)

    Đầu ra:
        np.ndarray: mảng giá trị F0 cho từng khung thời gian (đơn vị Hz)
    """
    # --- 1. Đọc âm thanh ---
    waveform, sr = load_audio_sf(audio_path)

    # --- 2. Áp dụng SwiftF0 (torchaudio) ---
    pitch = torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate=sr,
        frame_time=hop_length / sr,
        freq_low=fmin,
        freq_high=fmax
    )

    # --- 3. Xử lý khung im lặng ---
    pitch[pitch == 0] = np.nan  # thay 0 bằng NaN để dễ nhận diện vùng không có tín hiệu

    return pitch.squeeze().numpy()


def process_with_swiftf0(ref_path, user_path, output_dir="beat_eval_system/output"):
    """
    Thực hiện toàn bộ quy trình trích xuất pitch cho cả mẫu gốc và mẫu người dùng.

    Đầu vào:
        ref_path (str): đường dẫn tới tệp âm thanh gốc (original.wav)
        user_path (str): đường dẫn tới tệp thu của người dùng (record.wav)
        output_dir (str): thư mục lưu trữ kết quả (mặc định: 'beat_eval_system/output')

    Đầu ra:
        tuple (f0_ref, f0_user): 
            - f0_ref: cao độ mẫu gốc (np.ndarray)
            - f0_user: cao độ bản thu người dùng (np.ndarray)
    """
    print("🎵 Đang trích xuất cao độ (SwiftF0)...")

    os.makedirs(output_dir, exist_ok=True)
    f0_ref = extract_pitch_swift(ref_path)
    f0_user = extract_pitch_swift(user_path)

    # --- 4. Lưu kết quả ---
    np.save(os.path.join(output_dir, "f0_ref.npy"), f0_ref)
    np.save(os.path.join(output_dir, "f0_user.npy"), f0_user)

    print("✅ Trích xuất pitch hoàn tất. Kết quả được lưu tại thư mục output/: f0_ref.npy & f0_user.npy")

    return f0_ref, f0_user


# ============================================================
#  KIỂM THỬ MODULE TRỰC TIẾP
# ============================================================
if __name__ == "__main__":
    f0_ref, f0_user = process_with_swiftf0(
        "beat_eval_system/output/preprocessed_original.wav",
        "beat_eval_system/output/preprocessed_record.wav"
    )
    print("F0 mẫu gốc:", f0_ref[:10])
    print("F0 bản thu:", f0_user[:10])
