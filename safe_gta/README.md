# Safe-GTA — Baseline & Toy Task 程式碼說明

> 截止日：2026-05-08｜環境：`conda activate safe_gta`

---

## 這個 Project 在幹嘛？

我們要訓練一個「安全的自駕車 RL agent」，但現有的訓練資料裡有很多危險行為（亂切車道、衝出邊界）。

Safe-GTA 的做法：
1. 用 **Diffusion Model** 把爛資料「修復」成乾淨的軌跡
2. 修復時同時讓 **Safety Critic** 把關，確保修出來的路徑不危險
3. 拿修好的資料去訓練 **CQL（Offline RL）**

這個資料夾是第一階段：先用簡單的玩具任務驗證想法，跑出 Baseline 數字作為比較基準。

---

## 目錄結構

```
safe_gta/
├── diffusion/              # Diffusion Model 的核心元件（三個零件）
│   ├── noise_schedule.py
│   ├── unet1d.py
│   └── ddpm.py
├── toy_task1/              # Toy Task 1：驗證 Diffusion Model 能不能修軌跡
│   ├── data_gen.py
│   ├── train.py
│   └── evaluate.py
├── toy_task3_critic.py     # Toy Task 3：驗證 Safety Critic 判斷準不準
├── baseline2/
│   └── generate_augmented.py  # Baseline 2：不加安全限制的資料增強
├── train_cql.py            # Baseline 1：什麼都不改，直接跑 CQL
└── README.md               # 你正在看的這個
```

---

## 執行順序

```bash
# 1. 訓練 Diffusion Model（大概跑 5~10 分鐘）⚠️ 確認再跑
python -m safe_gta.toy_task1.train

# 2. 評估 Toy Task 1（會產生兩張圖）
python -m safe_gta.toy_task1.evaluate --start_t200

# 3. Toy Task 3 — Safety Critic
python -m safe_gta.toy_task3_critic

# 4. Baseline 1 — CQL
python -m safe_gta.train_cql

# 5. Baseline 2 — 生成不安全的增強資料（需先完成步驟 1）
python -m safe_gta.baseline2.generate_augmented
```

---

## 各檔案白話說明

---

### `diffusion/noise_schedule.py` — 「加噪聲 / 去噪聲」的數學規則

Diffusion Model 訓練的核心概念是：
- **訓練階段**：把乾淨資料一步一步加噪聲，讓 model 學會「噪聲長什麼樣子」
- **使用階段**：反過來，一步一步把噪聲去掉，還原成乾淨的軌跡

這個檔案定義的就是「每一步要加多少噪聲」的時間表（用 cosine 曲線，一開始加得少、最後加得多），以及兩個核心操作：

- `q_sample(x0, t)` → 把乾淨軌跡加噪到第 t 步，**訓練時用**
- `p_sample_step(model, x_t, t)` → 給模型，從 x_t 去噪一步，**使用時用**
- `p_sample_loop(...)` → 重複去噪很多步，完成完整修復

---

### `diffusion/unet1d.py` — 負責「預測噪聲」的神經網路

模型的本體。給它一條「帶噪聲的軌跡」和「現在是第幾步」，它會預測「這條軌跡裡有多少噪聲、噪聲長什麼樣子」。

架構是 **1D Temporal U-Net**，形狀像 U 字型：
- 左半部（Encoder）：把軌跡壓縮，抓出大方向的形狀特徵
- 底部（Bottleneck）：最濃縮的表示
- 右半部（Decoder）：展開還原，加回細節
- 橫向連線（Skip connections）：讓細節不會在壓縮時消失

> 就像 Photoshop 的「液化」工具，既能調整大方向的形狀，又能保留細節

參數量約 50 萬，在 GTX 1660 Ti 上幾分鐘就能跑完。

---

### `diffusion/ddpm.py` — 把上面兩個零件組裝成工具箱

前兩個檔案只是零件，這個檔案把它們組裝成可以直接使用的工具：

| 方法                           | 白話說明                                        |
| ---------------------------- | ------------------------------------------- |
| `training_loss(x0)`          | 拿一條乾淨軌跡，加噪聲，讓 model 預測噪聲，算預測的誤差（loss）       |
| `repair(x_bad, start_t=500)` | **SDEdit 修復**：把壞軌跡加噪到 t=500，再去噪回 0，「洗掉」壞的部分 |
| `generate(n)`                | 從純噪聲生成全新軌跡（Baseline 2 用，不修復，直接創造）           |
| `guided_repair(...)`         | 預留位置，之後加入 Safety Critic 引導（Safe-GTA 正式版用）   |

`start_t` 的直覺：越大 = 修復越激進，越小 = 只改細節

---

### `toy_task1/data_gen.py` — 製造訓練用的假資料

因為 MetaDrive 全套太重，先用自己造的假資料驗想法。

- **好軌跡**：一條直線，x 從 0 走到 10，y 保持在 0 附近（代表「正確地走在車道中間」）
- **壞軌跡**：好軌跡 + 隨機噪聲 + sin 波飄移（代表「走歪了、抖動的危險路徑」）

還負責把資料歸一化到 `[-1, 1]`，因為 Diffusion Model 對數值範圍很敏感，不歸一化的話 loss 會亂飛。

---

### `toy_task1/train.py` — 訓練 Diffusion Model

把上面造好的好軌跡丟進去，訓練 200 個 epoch，讓 model 學會「好的軌跡長什麼樣子」。

訓練設定：
- AdamW 優化器（比 Adam 更穩定）
- 學習率 2e-4，自動 warmup + 衰減
- T=200 時間步（比正式的 T=1000 快 5 倍，玩具任務夠用）

最後存下 checkpoint（`checkpoints/toy_task1_final.pt`）和 loss 曲線圖。

> ⚠️ 這個腳本是唯一要跑比較久的，跑之前先確認

---

### `toy_task1/evaluate.py` — 驗收 Toy Task 1

載入訓練好的 model，把壞軌跡丟進去修復，看修得好不好。

輸出三樣東西：

1. **數字**：修復前後各自距離「理想直線 y=0」的平均距離，必須 `repaired < bad`
2. **`repair_comparison.png`**：上排是壞的（紅色），下排是修復後（藍色），一眼看出效果
3. **`denoising_timelapse.png`**：從 t=500 到 t=0 的修復過程，顏色從紅漸漸變藍

合格標準：> 85% 的軌跡修復後比修復前更接近直線

---

### `toy_task3_critic.py` — 驗證 Safety Critic 判斷準不準

獨立測試「安全裁判」夠不夠格，不要等整個 pipeline 串起來才發現它根本判斷不準。

做法：手工造出一批「明顯安全」和「明顯危險」的狀態-動作對，然後看 Safety Critic 能不能正確分類：

- **安全**：在車道中間、速度正常、朝正確方向 → 預測 cost < 0.3
- **危險**：快撞牆了還繼續往牆衝、高速 → 預測 cost > 0.7

合格標準：正確率 > 80%

還會產生一張分布圖（`safety_critic_costs.png`），直觀看出兩群資料有沒有被明顯分開。

---

### `baseline2/generate_augmented.py` — 不管安不安全，先生一堆資料（Baseline 2）

用訓練好的 Diffusion Model 對 OSRL 的原始資料做 SDEdit 修復，生出更多軌跡。

**關鍵點：cost label 故意設為 0.0**，也就是不去判斷生出來的路徑安不安全，全部當成「沒問題的資料」丟進去訓練。

這麼做是為了製造一個對照組：
- Baseline 2 預期結果：return 高（資料多了，分數上去），但 constraint violation 也高（因為沒有安全過濾）
- 對比 Safe-GTA（有 Safety Critic 引導）的結果，就能看出「加安全限制到底差多少」

---

### `train_cql.py` — 什麼都不改，直接跑 CQL（Baseline 1）

最簡單的對照組：用原始的爛資料集直接跑 Conservative Q-Learning，不做任何資料處理。

CQL 的特色是在一般 Q-learning 上多加一個「保守懲罰項」，讓 model 不要對沒見過的動作過度樂觀估計（這是 Offline RL 的常見問題）。

最後輸出三個指標存成 JSON：

| 指標 | 意思 |
|------|------|
| Normalized Return | 跟專家比，得了幾分（越高越好） |
| Safety Success Rate | 跑完全程沒出事的比例（越高越好） |
| Constraint Violation Cost | 累積撞牆/出界的罰分（越低越好） |

這三個數字就是 Safe-GTA 要打敗的「下限」。

---

## 驗收 Checklist

- [ ] Toy Task 1：`results/repair_comparison.png` 看得出修復效果
- [ ] Toy Task 1：repaired L2 < bad L2（> 85% 的軌跡有改善）
- [ ] Toy Task 3：Safety Critic 準確率 > 80%
- [ ] Baseline 1：`results/baseline1_cql_results.json` 有三個數字
- [ ] Baseline 2：`data/augmented_no_safety.pkl` 有生出來
