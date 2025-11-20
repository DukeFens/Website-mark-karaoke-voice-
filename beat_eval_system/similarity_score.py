# similarity_score.py

"""
Module: similarity_score
Chức năng: Tính % similarity giữa bản thu (record.wav) và bản gốc (original.wav)
Dựa trên kết quả từ alignment & metric computation.
"""

import numpy as np

def compute_similarity(metrics, weights=None, scales=None):
    """
    Tính toán % similarity dựa trên các chỉ số từ result_metrics.

    Tham số
    --------
    metrics : dict
        Kết quả từ run_alignment_metric (result_metrics)
    weights : dict, optional
        Trọng số cho từng metric (pitch, timing, energy, formant, rhythmic)
    scales : dict, optional
        Giá trị chuẩn hóa/scale tối đa cho từng metric để chuyển sang score 0-1

    Trả về
    -------
    float
        Giá trị similarity (%) từ 0 đến 100
    """

    # --- 1. Đặt trọng số mặc định ---
    if weights is None:
        weights = {
            "pitch": 0.35,
            "timing": 0.25,
            "onbeat": 0.10,
            "energy": 0.10,
            "formant": 0.10,
            "rhythmic": 0.10
        }

    # --- 2. Đặt scale chuẩn hóa tối đa ---
    # Các giá trị này dùng để chuyển deviation -> 0..1 score
    if scales is None:
        scales = {
            "pitch": 600,         # cent
            "timing": 500,        # ms
            "onbeat": 100,        # %
            "energy": 0.2,        # RMS diff
            "formant": 10,        # MFCC diff
            "rhythmic": 1.0       # std IOI
        }

    # --- 3. Tính score cho từng metric ---
    pitch_score = max(0, 1 - (metrics["pitch_deviation_mean"] / scales["pitch"])) if metrics["pitch_deviation_mean"] is not None else 0
    timing_score = max(0, 1 - (metrics["timing_deviation"] / scales["timing"])) if metrics["timing_deviation"] is not None else 0
    onbeat_score = (metrics["onbeat_accuracy"] / scales["onbeat"]) if metrics["onbeat_accuracy"] is not None else 0
    energy_score = max(0, 1 - (metrics["energy_deviation_mean"] / scales["energy"])) if metrics["energy_deviation_mean"] is not None else 0
    formant_score = max(0, 1 - (metrics["formant_stability"] / scales["formant"])) if metrics["formant_stability"] is not None else 0
    rhythmic_score = max(0, 1 - (metrics["rhythmic_variation"] / scales["rhythmic"])) if metrics["rhythmic_variation"] is not None else 0

    # --- 4. Tính weighted sum ---
    similarity = (
        pitch_score * weights["pitch"] +
        timing_score * weights["timing"] +
        onbeat_score * weights["onbeat"] +
        energy_score * weights["energy"] +
        formant_score * weights["formant"] +
        rhythmic_score * weights["rhythmic"]
    )

    # --- 5. Chuyển sang %
    similarity_percent = float(similarity * 100.0)
    similarity_percent = min(max(similarity_percent, 0.0), 100.0)

    return similarity_percent


# === KIỂM TRA MODULE ===
if __name__ == "__main__":
    # Ví dụ thử nghiệm
    example_metrics = {
        "pitch_deviation_mean": 284.8,
        "timing_deviation": 217.0,
        "onbeat_accuracy": 10.0,
        "energy_deviation_mean": 0.0468,
        "formant_stability": 5.44,
        "rhythmic_variation": 0.61
    }
    sim = compute_similarity(example_metrics)
    print(f"Similarity: {sim:.2f}%")
