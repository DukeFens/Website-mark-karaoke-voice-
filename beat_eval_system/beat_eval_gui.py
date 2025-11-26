import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import json

# ==== Import pipeline functions ====
from main_pipeline import ensure_wav_format, preprocess_audio
from beat_transformer_module import process_with_beat_transformer
from swift_f0_module import process_with_swiftf0
from alignment_metric_module import run_alignment_metric
from similarity_score import compute_similarity
import soundfile as sf

# ==== GUI Class ====
class BeatEvalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Beat & Pitch Evaluation System")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        # Paths
        self.original_path = None
        self.record_path = None

        # ==== GUI Elements ====
        tk.Label(root, text="🎧 Original Audio:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.orig_entry = tk.Entry(root, width=50)
        self.orig_entry.grid(row=0, column=1, padx=10)
        tk.Button(root, text="Browse", command=self.browse_original).grid(row=0, column=2, padx=5)

        tk.Label(root, text="🎤 Recorded Audio:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.rec_entry = tk.Entry(root, width=50)
        self.rec_entry.grid(row=1, column=1, padx=10)
        tk.Button(root, text="Browse", command=self.browse_record).grid(row=1, column=2, padx=5)

        self.run_button = tk.Button(root, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.grid(row=2, column=1, pady=10)

        tk.Label(root, text="📊 Metrics Output:").grid(row=3, column=0, sticky="nw", padx=10)
        self.text_output = scrolledtext.ScrolledText(root, width=80, height=20)
        self.text_output.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

    # ==== Browse Functions ====
    def browse_original(self):
        path = filedialog.askopenfilename(title="Select Original Audio", filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")])
        if path:
            self.original_path = path
            self.orig_entry.delete(0, tk.END)
            self.orig_entry.insert(0, path)

    def browse_record(self):
        path = filedialog.askopenfilename(title="Select Recorded Audio", filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")])
        if path:
            self.record_path = path
            self.rec_entry.delete(0, tk.END)
            self.rec_entry.insert(0, path)

    # ==== Run Pipeline ====
    def run_pipeline(self):
        if not self.original_path or not self.record_path:
            messagebox.showwarning("⚠️ Missing Files", "Please select both original and recorded audio files.")
            return

        try:
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, "⏳ Running pipeline...\n")
            self.root.update()

            # --- Ensure WAV ---
            orig_path = ensure_wav_format(self.original_path)
            rec_path = ensure_wav_format(self.record_path)

            # --- Preprocess ---
            y_ref, sr = preprocess_audio(orig_path)
            y_user, sr = preprocess_audio(rec_path)

            # Save preprocessed
            os.makedirs("beat_eval_system/output", exist_ok=True)
            orig_pre = "beat_eval_system/output/preprocessed_original.wav"
            rec_pre = "beat_eval_system/output/preprocessed_record.wav"
            sf.write(orig_pre, y_ref, sr)
            sf.write(rec_pre, y_user, sr)

            # --- Beat Transformer ---
            beat_output = process_with_beat_transformer(orig_pre)

            # --- SwiftF0 ---
            f0_ref, f0_user = process_with_swiftf0(orig_pre, rec_pre)

            # --- Alignment & Metric ---
            metrics = run_alignment_metric(f0_user, f0_ref, y_user=y_user, y_ref=y_ref, beat_times=beat_output["beat_times"])

            # --- Compute similarity %
            similarity_percent = compute_similarity(metrics)
            metrics["similarity_percent"] = similarity_percent

            # --- Save JSON ---
            output_path = "beat_eval_system/output/result.json"
            def np_convert(o):
                import numpy as np
                if isinstance(o, (np.float32, np.float64, np.int32, np.int64, np.integer)):
                    return o.item()
                if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
                    return None
                return str(o)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, default=np_convert)

            # --- Display in GUI ---
            self.text_output.insert(tk.END, "✅ Pipeline completed!\n")
            self.text_output.insert(tk.END, f"Result can be found at: {output_path}\n\n")
            self.text_output.insert(tk.END, json.dumps(metrics, indent=4, default=np_convert))
            self.text_output.insert(tk.END, f"\n\n🎯 Similarity: {similarity_percent:.2f}%\n")

        except Exception as e:
            messagebox.showerror("❌ Error", str(e))
            self.text_output.insert(tk.END, f"Error: {e}\n")


# ==== Run App ====
if __name__ == "__main__":
    root = tk.Tk()
    app = BeatEvalGUI(root)
    root.mainloop()
