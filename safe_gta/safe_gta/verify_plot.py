import numpy as np
import matplotlib.pyplot as plt

def plot_verification():
    print("📊 正在載入資料進行視覺化驗證...")
    
    # 讀取剛切好的資料
    good_data = np.load("data/processed/metadrive_good.npz", allow_pickle=True)['data']
    bad_data = np.load("data/processed/metadrive_bad.npz", allow_pickle=True)['data']
    
    # 抽出第一筆來檢查
    good_sample = good_data[0]
    bad_sample = bad_data[0]
    
    # 設定畫布
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('MetaDrive Trajectory Verification: Good vs Bad', fontsize=16)
    
    # --- 繪製 Good 軌跡 ---
    axs[0, 0].set_title('Good Trajectory: Actions (Steering & Accel)')
    axs[0, 0].plot(good_sample['actions'][:, 0], label='Steering', color='blue', alpha=0.7)
    axs[0, 0].plot(good_sample['actions'][:, 1], label='Acceleration', color='green', alpha=0.7)
    axs[0, 0].legend()
    
    axs[1, 0].set_title('Good Trajectory: Cost (Should be flat 0)')
    axs[1, 0].plot(good_sample['costs'], color='red')
    axs[1, 0].set_ylim(-0.1, 1.1) # Cost 通常是 0 或 1
    
    # --- 繪製 Bad 軌跡 ---
    axs[0, 1].set_title('Bad Trajectory: Actions (Steering & Accel)')
    axs[0, 1].plot(bad_sample['actions'][:, 0], label='Steering', color='blue', alpha=0.7)
    axs[0, 1].plot(bad_sample['actions'][:, 1], label='Acceleration', color='green', alpha=0.7)
    axs[0, 1].legend()
    
    axs[1, 1].set_title('Bad Trajectory: Cost (Notice the crash at the end)')
    axs[1, 1].plot(bad_sample['costs'], color='red')
    axs[1, 1].set_ylim(-0.1, max(bad_sample['costs']) + 0.5)
    
    plt.tight_layout()
    plt.savefig("verification_plot.png")
    print("✅ 圖表已儲存為 verification_plot.png，請打開來看看吧！")

if __name__ == "__main__":
    plot_verification()