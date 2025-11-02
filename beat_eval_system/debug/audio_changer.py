import librosa, soundfile as sf

y, sr = librosa.load("beat_eval_system/input/original.wav", sr=16000)

# Lệch cao độ xuống 1 semitone (~ -6%)
y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)

# Giảm tốc độ 5%
D = librosa.stft(y_pitch)
D_stretch = librosa.phase_vocoder(D, rate=0.95, hop_length=512)
y_stretch = librosa.istft(D_stretch)

sf.write("beat_eval_system/input/record.wav", y_stretch, sr)
print("✅ record.wav đã được tạo thành công.")
