"""
feedback_module.py

Mô-đun sinh phản hồi luyện hát dựa trên các chỉ số đánh giá từ Beat Evaluation System.

Mô-đun gửi prompt đến LLM (Groq) và yêu cầu phản hồi ở dạng JSON với
đúng 3 trường: "pitch", "timing", "energy".

Không in log ra console. Mọi lỗi được raise bình thường.
"""

import json
from groq import Groq


# =============================================================================
# DIRECT EMBEDDED API KEY
# =============================================================================
GROQ_API_KEY = "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"


# =============================================================================
# PROMPT CONSTRUCTION (with full music metric knowledge embedded)
# =============================================================================
def build_prompt(result_json_text):
    """Xây dựng prompt gửi cho LLM.

    Embed toàn bộ hướng dẫn chuyên sâu để LLM hiểu đúng ý nghĩa tất cả
    các thông số trong result.json, dù đầu ra chỉ có 3 trường.

    Args:
        result_json_text (str): JSON string chứa toàn bộ metric từ pipeline.

    Returns:
        str: Prompt hoàn chỉnh gửi cho mô hình.
    """
    return f"""
Bạn là trợ lý đánh giá giọng hát. Nhiệm vụ của bạn là đọc các metric kỹ thuật
trong JSON và tạo phản hồi ngắn gọn để giúp người hát cải thiện.

=========================
 HƯỚNG DẪN HIỂU TOÀN BỘ METRIC
=========================

**1. Pitch-related metrics**
- pitch_deviation_mean: độ lệch trung bình so với pitch chuẩn.
- pitch_deviation_std: độ dao động pitch.
- f0_accuracy: mức độ hát trúng nốt theo time-aligned F0.
- formant_stability: độ ổn định cộng hưởng âm sắc (ảnh hưởng độ rõ và tròn tiếng).
- vibrato_rate / vibrato_extent: độ rung giọng.
→ Pitch không chỉ là “trúng nốt”, mà là khả năng giữ cao độ ổn định, không trôi nốt, không run thất thường.

**2. Timing-related metrics**
- timing_deviation: sai số thời điểm vào câu so với nhịp chuẩn.
- onbeat_accuracy: % lần hát đúng trên beat.
- onset_strength / microtiming_variation: độ sắc nét khi bắt âm.
- rhythmic_variation: mức lệch nhịp nhỏ giữa các nốt.
→ Timing không chỉ là vào đúng nhịp, mà còn bao gồm cảm giác nhạc, độ chắc của onset.

**3. Energy / Dynamics metrics**
- energy_deviation_mean / std: thay đổi năng lượng giọng.
- dynamics_curve: cách phân bố lực giọng theo câu.
- spectral_flatness / brightness: ảnh hưởng độ rõ.
→ Energy không chỉ là hát mạnh – yếu mà là kiểm soát hơi, độ đều, tránh bị hụt hơi.

**4. Tone quality / Timbre metrics**
- formant shifts: biến đổi màu giọng.
- harmonicity: độ sạch của nguồn thanh.
- noise components: hơi thở, xì.
→ Ảnh hưởng trực tiếp tới cảm giác “hát hay” hoặc “bí – tối – nghẹt”.

**5. Other alignment metrics**
- DTW alignment quality
- note-level mapping consistency

-------------------------
 YÊU CẦU QUAN TRỌNG
-------------------------
✓ Feedback phải ngắn gọn, tiếng Việt đơn giản.  
✓ Chỉ nói điều cần cải thiện, không khen.  
✓ Mỗi trường 1–2 câu ngắn, hành động cụ thể.  
✓ KHÔNG nhắc lại số liệu.  
✓ KHÔNG giải thích metric.  
✓ Đầu ra PHẢI là JSON với đúng 3 trường:
  "pitch", "timing", "energy"

-------------------------
 DỮ LIỆU EVALUATION
-------------------------
{result_json_text}

Bây giờ hãy trả về JSON phản hồi.
"""


# =============================================================================
# REQUEST
# =============================================================================
def request_feedback(prompt):
    """Gửi prompt đến Groq LLM và trả về JSON đã parse.

    Args:
        prompt (str): Nội dung prompt.

    Returns:
        dict: Object gồm 3 trường:
              - pitch
              - timing
              - energy

    Raises:
        ValueError: JSON trả về không hợp lệ.
        Exception: Mọi lỗi khác từ mô hình.
    """
    client = Groq(api_key=GROQ_API_KEY)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=4096,
        top_p=0.95,
        reasoning_effort="high",
        response_format={"type": "json_object"},
        stream=False,
    )

    response_text = completion.choices[0].message.content
    feedback = json.loads(response_text)

    # Ensure required keys exist even if model omits them
    for key in ["pitch", "timing", "energy"]:
        feedback.setdefault(key, "")

    return feedback


# =============================================================================
# MAIN ENTRY
# =============================================================================
def generate_feedback(result_metrics):
    """Sinh phản hồi hát từ metrics của pipeline.

    Args:
        result_metrics (dict): Toàn bộ metric từ Beat Evaluation System.

    Returns:
        dict: Object JSON gồm 3 trường: pitch, timing, energy.
    """
    result_text = json.dumps(result_metrics, ensure_ascii=False)
    prompt = build_prompt(result_text)
    return request_feedback(prompt)


if __name__ == "__main__":
    pass
