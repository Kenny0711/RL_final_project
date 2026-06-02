import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# 1. 神經網路架構 (Safety Critic)
# ==========================================
class SafetyCritic(nn.Module):
    def __init__(self, state_dim=259, action_dim=2, hidden_dim=256):
        super(SafetyCritic, self).__init__()
        # 狀態 (259) + 動作 (2) = 261 維輸入
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # 👈 新增：隨機丟棄 20% 的神經元防止死背
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # 👈 新增
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        # 將 state 和 action 拼在一起餵給神經網路
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# ==========================================
# 2. 資料讀取與處理
# ==========================================
def load_and_prepare_data():
    print("🛠️ 正在載入真實 MetaDrive 軌跡資料...")
    good_data = np.load("data/processed/metadrive_good.npz", allow_pickle=True)['data']
    bad_data = np.load("data/processed/metadrive_bad.npz", allow_pickle=True)['data']

    safe_X, safe_Y = [], []
    dang_X, dang_Y = [], []

    # 處理 Safe Data (標籤為 0)
    for traj in good_data:
        obs = traj['observations']
        acts = traj['actions']
        for i in range(len(obs)):
            safe_X.append(np.concatenate([obs[i], acts[i]]))
            safe_Y.append([0.0]) # 0 代表安全

    # 處理 Dangerous Data (標籤為 1)
    for traj in bad_data:
        obs = traj['observations']
        acts = traj['actions']
        for i in range(len(obs)):
            dang_X.append(np.concatenate([obs[i], acts[i]]))
            dang_Y.append([1.0]) # 1 代表危險

    safe_X = np.array(safe_X, dtype=np.float32)
    dang_X = np.array(dang_X, dtype=np.float32)
    
    print(f"📊 原始資料量 -> 安全步數: {len(safe_X)}, 危險步數: {len(dang_X)}")

    # 【關鍵防護】平衡資料量 (Undersampling)
    # 危險步數通常遠多於安全步數，我們把危險步數隨機抽樣，降到跟安全步數一樣多
    min_len = min(len(safe_X), len(dang_X))
    np.random.seed(42) # 固定亂數種子，確保每次訓練結果可重現
    dang_indices = np.random.choice(len(dang_X), min_len, replace=False)
    
    safe_X = safe_X[:min_len]
    dang_X = dang_X[dang_indices]
    
    safe_Y = np.zeros((min_len, 1), dtype=np.float32)
    dang_Y = np.ones((min_len, 1), dtype=np.float32)

    # 合併並打亂資料
    X = np.vstack([safe_X, dang_X])
    Y = np.vstack([safe_Y, dang_Y])
    
    indices = np.random.permutation(len(X))
    X, Y = X[indices], Y[indices]

    print(f"⚖️ 平衡後資料量 -> 總共 {len(X)} 筆 (Safe {min_len} 筆, Dangerous {min_len} 筆)")
    return X, Y

# ==========================================
# 3. 訓練迴圈
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用硬體加速: {device}")

    # 準備資料
    X, Y = load_and_prepare_data()
    
    # 切割 Train (80%) 與 Val (20%)
    split_idx = int(len(X) * 0.8)
    X_train, Y_train = torch.FloatTensor(X[:split_idx]), torch.FloatTensor(Y[:split_idx])
    X_val, Y_val = torch.FloatTensor(X[split_idx:]), torch.FloatTensor(Y[split_idx:])

    # 建立 DataLoader
    batch_size = 256
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、Loss 與優化器
    model = SafetyCritic().to(device)
    criterion = nn.BCELoss() # 二元交叉熵 (適合 0~1 的分類)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val_acc = 0.0
    print("\n🔥 開始訓練 Safety Critic...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 拆分 State 和 Action (前 259 是 State, 後 2 是 Action)
            states = batch_x[:, :259]
            actions = batch_x[:, 259:]
            
            # 前向傳播
            preds = model(states, actions)
            loss = criterion(preds, batch_y)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 驗證階段 (Validation)
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                states = batch_x[:, :259]
                actions = batch_x[:, 259:]
                
                preds = model(states, actions)
                val_loss += criterion(preds, batch_y).item()
                
                # 計算準確率 (機率 > 0.5 視為預測危險)
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == batch_y).sum().item()
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / len(X_val) * 100

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # 儲存訓練好的模型
        if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                os.makedirs("data/processed", exist_ok=True)
                save_path = "data/processed/real_safety_critic.pt"
                torch.save(model.state_dict(), save_path)
                print(f"   🌟 破紀錄了！儲存當前最佳模型 (Acc: {best_val_acc:.2f}%)")
    print(f"\n🎉 訓練完成！歷史最佳模型準確率: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()