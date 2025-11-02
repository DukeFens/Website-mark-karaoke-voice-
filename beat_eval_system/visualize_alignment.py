import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dtw import dtw

# ==== 1. Load F0 ====
f0_ref = np.load("beat_eval_system/output/f0_ref.npy")
f0_user = np.load("beat_eval_system/output/f0_user.npy")

# ==== 2. Chuẩn hóa thời gian ====
t_ref = np.linspace(0, len(f0_ref)/100, len(f0_ref))
t_user = np.linspace(0, len(f0_user)/100, len(f0_user))

# ==== 3. Vẽ đường cong pitch ====
plt.figure(figsize=(10,4))
plt.plot(t_ref, f0_ref, label="Original (Ref)", alpha=0.8)
plt.plot(t_user, f0_user, label="Record (User)", alpha=0.8)
plt.title("Pitch Contours")
plt.xlabel("Time (s)")
plt.ylabel("F0 (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== 4. Tính DTW Path ====
dist, cost, acc_cost, path = dtw(
    f0_user.reshape(-1,1),
    f0_ref.reshape(-1,1),
    dist=lambda x, y: np.linalg.norm(x - y)
)

print(f"✅ DTW distance = {dist:.2f}")

# ==== 5. Vẽ đường căn chỉnh (DTW path) ====
plt.figure(figsize=(6,6))
plt.imshow(acc_cost.T, origin='lower', cmap='magma', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.title("DTW Path between User & Ref F0")
plt.xlabel("User (record.wav)")
plt.ylabel("Reference (original.wav)")
plt.colorbar(label="Accumulated Cost")
plt.tight_layout()
plt.show()
