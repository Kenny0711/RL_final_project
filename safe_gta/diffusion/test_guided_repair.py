import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from diffusion.ddpm import DDPM
from diffusion.noise_schedule import NoiseSchedule
from diffusion.unet1d import TemporalUNet
# 匯入隊友的資料載入模組
from toy_task1.data_gen import load_dataset, denormalize_trajectories

# ==========================================
# 1. 建立簡單的 Dummy Safety Critic
# 假設安全區域是經過去噪還原、去標準化後的 y 軸在 -0.5 ~ 0.5 之間
# ==========================================
class DummySafetyCritic(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        # ----------------------------------------------------
        # 【安全修正】：不管 stats 的維度是多少，用 .flatten() 
        # 直接確保它變成一維，然後抓最後一個元素（通常是 Y 軸）
        # ----------------------------------------------------
        y_min_val = stats["min"].flatten()[-1]
        y_max_val = stats["max"].flatten()[-1]
        
        print(f"[Debug] Detected normalization stats -> Y_Min: {y_min_val}, Y_Max: {y_max_val}")
        
        self.y_min = float(y_min_val)
        self.y_max = float(y_max_val)

    def forward(self, x_t):
        # x_t shape: (B, Seq_Len, 2)
        y_norm = x_t[..., 1]
        
        # 反標準化回真實 Y 座標
        y_raw = y_norm * (self.y_max - self.y_min) + self.y_min
        
        # 假設在真實世界中，只要 y 絕對值大於 0.5 就是衝出賽道
        upper_bound = 0.5
        lower_bound = -0.5
        
        cost_upper = torch.relu(y_raw - upper_bound) ** 2
        cost_lower = torch.relu(lower_bound - y_raw) ** 2
        return (cost_upper + cost_lower).sum(-1)

# ==========================================
# 2. 測試主程式
# ==========================================
def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on {device}...")

    # A. 載入 Toy Task 1 生成的真實資料分佈與標準化常數
    data_path = "safe_gta/data/toy_task1.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到隊友的假資料 {data_path}，請先執行 python -m toy_task1.data_gen")
        
    good_norm, bad_norm, stats = load_dataset(data_path)

    # 抽出一條真正蛇行、撞牆的真實爛資料作為測試目標
    x_bad_norm = bad_norm[0:1] # shape: (1, 50, 2)

    # B. 初始化模型架構
    seq_len = 50
    n_features = 2
    schedule = NoiseSchedule(T=200, schedule="cosine", device=device)
    unet = TemporalUNet(seq_len=seq_len, n_features=n_features, base_ch=32).to(device)
    ddpm = DDPM(model=unet, schedule=schedule, device=device)

    # C. 載入真實訓練好的權重
    ckpt_path = "checkpoints/diffusion.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading trained weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state" in checkpoint:
            unet.load_state_dict(checkpoint["model_state"])
        else:
            unet.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"找不到權重檔 {ckpt_path}！")
    
    unet.eval()

    safety_critic = DummySafetyCritic(stats).to(device)

    print("Starting Trajectory Repair...")
    test_start_t = 70

    # --- 測試 1：沒有安全引導 (beta = 0.0) ---
    print("Test 1: Repairing with beta = 0.0 (Pure Denoise)...")
    repaired_no_safety_norm = ddpm.guided_repair(
        x_bad=x_bad_norm, safety_critic=safety_critic, beta=0.0, start_t=test_start_t
    )

    # --- 測試 2：啟動安全引導 (有梯度截斷保護，beta 給 1.0 非常安全) ---
    print("Test 2: Repairing with beta = 1.0 (Physics-Informed Guidance)...")
    repaired_safe_norm = ddpm.guided_repair(
        x_bad=x_bad_norm, safety_critic=safety_critic, beta=20.0, start_t=test_start_t
    )

    # D. 【最關鍵：要把成果「反標準化」回到真實世界座標才能畫出正確的圖】
    x_bad_raw = denormalize_trajectories(x_bad_norm, stats)[0]
    repaired_no_safety_raw = denormalize_trajectories(repaired_no_safety_norm, stats)[0]
    repaired_safe_raw = denormalize_trajectories(repaired_safe_norm, stats)[0]

    # E. 畫圖驗證
    plt.figure(figsize=(10, 5))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Upper Boundary (y=0.5)')
    plt.axhline(y=-0.5, color='r', linestyle='--', label='Lower Boundary (y=-0.5)')
    plt.fill_between(x_bad_raw[:, 0], -0.5, 0.5, color='green', alpha=0.1, label='Safe Zone')

    plt.plot(x_bad_raw[:, 0], x_bad_raw[:, 1], 'k-.', label='Original Unsafe (Bad)', alpha=0.6)
    plt.plot(repaired_no_safety_raw[:, 0], repaired_no_safety_raw[:, 1], 'orange', label='Pure Repair (Beta=0)', linewidth=2)
    plt.plot(repaired_safe_raw[:, 0], repaired_safe_raw[:, 1], 'blue', label='Safe-GTA Repair (Beta=1.0)', linewidth=2)

    plt.title("Safe-GTA: Fixed Guided Repair Verification")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.ylim(-2, 2)
    plt.legend()
    plt.grid(True)
    
    os.makedirs("safe_gta/results", exist_ok=True)
    out_path = "safe_gta/results/guided_repair_test.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\n[⚠️ 圖片底加啦 ⚠️] 絕對路徑是：\n{os.path.abspath(out_path)}")

if __name__ == "__main__":
    run_test()