import torch
import numpy as np
import os
import soundfile as sf
import torchaudio

# ==== SwiftF0: Fast and robust pitch extractor ====
# Reference: https://github.com/w-okada/voice-changer/blob/main/Realtime-Voice-Clone/swift_pitch_extraction.py

def load_audio_sf(audio_path):
    """
    Load WAV using soundfile (sf) to avoid TorchCodec issues on Windows.
    Returns waveform tensor [1, num_samples] and sample rate.
    """
    y, sr = sf.read(audio_path)
    if y.ndim > 1:  # stereo -> mono
        y = np.mean(y, axis=1)
    waveform = torch.from_numpy(y).unsqueeze(0).float()
    return waveform, sr

def extract_pitch_swift(audio_path, hop_length=160, fmin=50.0, fmax=1100.0):
    # Load audio (soundfile backend)
    waveform, sr = load_audio_sf(audio_path)

    # Apply Swift pitch extractor from torchaudio
    pitch = torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate=sr,
        frame_time=hop_length / sr,
        freq_low=fmin,
        freq_high=fmax
    )

    # Replace 0 with NaN (for silent frames)
    pitch[pitch == 0] = np.nan
    return pitch.squeeze().numpy()

def process_with_swiftf0(ref_path, user_path, output_dir="beat_eval_system/output"):
    print("🎵 Đang trích xuất pitch (SwiftF0)...")

    os.makedirs(output_dir, exist_ok=True)
    f0_ref = extract_pitch_swift(ref_path)
    f0_user = extract_pitch_swift(user_path)

    np.save(os.path.join(output_dir, "f0_ref.npy"), f0_ref)
    np.save(os.path.join(output_dir, "f0_user.npy"), f0_user)

    print("✅ Pitch extraction hoàn tất. File lưu trong output/: f0_ref.npy & f0_user.npy")
    return f0_ref, f0_user
