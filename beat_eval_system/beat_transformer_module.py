import torch
import librosa
import numpy as np
from madmom.features import DBNBeatTrackingProcessor, RNNBeatProcessor

# ===== Beat Transformer module =====
def process_with_beat_transformer(audio_path):
    """
    Input:  preprocessed_original.wav
    Output: beat_times (s), downbeat_flags, tempo (BPM)
    """
    # --- 1. Load audio ---
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # --- 2. Model inference (RNN/Transformer-based) ---
    proc = RNNBeatProcessor()
    beat_act = proc(str(audio_path))

    # --- 3. Temporal decoding (DBN refinement) ---
    tracker = DBNBeatTrackingProcessor(fps=100)
    beat_times = tracker(beat_act)

    # --- 4. Derive tempo ---
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        tempo = 60.0 / np.median(intervals)
    else:
        tempo = 0

    # --- 5. Downbeat flags (simple heuristic) ---
    downbeat_flags = np.zeros(len(beat_times))
    downbeat_flags[::4] = 1  # every 4th beat as downbeat (4/4 assumption)

    return {
        "beat_times": beat_times.tolist(),
        "downbeat_flags": downbeat_flags.tolist(),
        "tempo": float(tempo)
    }

if __name__ == "__main__":
    out = process_with_beat_transformer("beat_eval_system/output/preprocessed_original.wav")
    print(out)
