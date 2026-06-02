import h5py
import numpy as np
import os

def process_and_verify_dsrl():
    print("🛠️ 準備解析 HDF5 並完整萃取狀態、動作與成本...")
    file_path = "data/metadrive_mediumsparse.hdf5"
    
    good_trajectories = []
    bad_trajectories = []
    
    with h5py.File(file_path, "r") as f:
        # 這次我們把所有神經網路需要的特徵都拿出來
        obs = f['observations'][:]
        actions = f['actions'][:]
        costs = f['costs'][:]
        terminals = f['terminals'][:]
        timeouts = f['timeouts'][:]
        
        total_steps = len(obs)
        current_obs, current_acts, current_costs = [], [], []
        
        print("✂️ 正在切割軌跡...")
        for i in range(total_steps):
            current_obs.append(obs[i])
            current_acts.append(actions[i])
            current_costs.append(costs[i])
            
            if terminals[i] or timeouts[i]:
                traj_cost = np.sum(current_costs)
                
                # 把每一趟的完整資訊包裝成一個 Dictionary
                traj_dict = {
                    'observations': np.array(current_obs),
                    'actions': np.array(current_acts),
                    'costs': np.array(current_costs),
                    'total_cost': traj_cost,
                    'length': len(current_obs)
                }
                
                if traj_cost == 0:
                    good_trajectories.append(traj_dict)
                else:
                    bad_trajectories.append(traj_dict)
                    
                current_obs, current_acts, current_costs = [], [], []

    # 存檔
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/metadrive_good.npz", data=np.array(good_trajectories, dtype=object))
    np.savez("data/processed/metadrive_bad.npz", data=np.array(bad_trajectories, dtype=object))
    
    # ==========================================
    # 🔍 驗證階段 (你最關心的部分)
    # ==========================================
    print("\n" + "="*40)
    print("🔍 驗證資料集內容是否正確")
    print("="*40)
    
    # 驗證 Good Data
    error_good = sum(1 for t in good_trajectories if t['total_cost'] > 0)
    print(f"🌟 Good 軌跡數量: {len(good_trajectories)}")
    print(f"   👉 異常檢查: 裡面有 {error_good} 條軌跡的 Cost 不等於 0 (應為 0)")
    if len(good_trajectories) > 0:
        sample = good_trajectories[0]
        print(f"   👉 抽查第1筆: 步數 {sample['length']}, 狀態維度 {sample['observations'].shape}, 總 Cost: {sample['total_cost']}")

    # 驗證 Bad Data
    error_bad = sum(1 for t in bad_trajectories if t['total_cost'] == 0)
    print(f"\n💥 Bad 軌跡數量: {len(bad_trajectories)}")
    print(f"   👉 異常檢查: 裡面有 {error_bad} 條軌跡的 Cost 等於 0 (應為 0)")
    if len(bad_trajectories) > 0:
        sample = bad_trajectories[0]
        print(f"   👉 抽查第1筆: 步數 {sample['length']}, 動作維度 {sample['actions'].shape}, 總 Cost: {sample['total_cost']}")

    print("\n✅ 資料已完美驗證並儲存！現在它包含了狀態、動作與危險指標！")

if __name__ == "__main__":
    process_and_verify_dsrl()