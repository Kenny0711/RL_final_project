# Safe-GTA Code 說明

這份文件是給隊友看的，目標是讓大家知道目前留下來的 code 各自負責什麼、要怎麼跑、會產生什麼結果。

目前只保留和 MetaDrive / Safe-GTA 主流程有關的 code，以及會產生下列資料夾中保留圖片或影片的 code：

```text
safe_gta/results/figure/
safe_gta/results/final_figures/
```

## 環境

```bash
conda activate safe_gta
cd "C:\Users\kenny\Desktop\School\nycu course\RL\Lab\final project"
```

## 主流程順序

### 1. 下載資料

```bash
python download_dataset.py
```

產出：

```text
safe_gta/data/metadrive_mediumsparse.hdf5
safe_gta/data/metadrive_mediummean.hdf5
safe_gta/data/metadrive_mediumdense.hdf5
```

### 2. 訓練 Diffusion Model

```bash
python -m safe_gta.train_diffusion --epochs 200 --batch_size 64
```

產出：

```text
safe_gta/checkpoints/diffusion_metadrive_final.pt
```

### 3. 訓練 Safety Critic

```bash
python -m safe_gta.safety_critic --epochs 20 --max_samples 200000
```

產出：

```text
safe_gta/checkpoints/metadrive_safety_critic.pt
safe_gta/results/metrics/metadrive_safety_critic_results.json
```

### 4. Baseline 1：Raw CQL

```bash
python -m safe_gta.train_cql --epochs 50
```

產出：

```text
safe_gta/checkpoints/baseline1_cql.pt
safe_gta/results/metrics/baseline1_cql_results.json
```

### 5. Baseline 2：CQL + GTA without Safety Critic

```bash
python -m safe_gta.baseline2.generate_augmented --checkpoint safe_gta/checkpoints/diffusion_metadrive_final.pt --n_augmented 5000 --start_t 150 --output safe_gta/data/augmented_no_safety.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl --epochs 50
```

產出：

```text
safe_gta/data/augmented_no_safety.pkl
safe_gta/checkpoints/baseline2_cql.pt
safe_gta/results/metrics/baseline2_cql_results.json
```

### 6. Safe-GTA：CQL + GTA + Safety Critic

目前建議參數：

```text
start_t = 25
beta = 2.0
```

```bash
python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 2.0 --start_t 25 --output safe_gta/data/augmented_safe_gta_t25_beta20.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_safe_gta_t25_beta20.pkl --epochs 50
```

產出：

```text
safe_gta/data/augmented_safe_gta_t25_beta20.pkl
safe_gta/checkpoints/safe_gta_cql.pt
safe_gta/results/metrics/safe_gta_cql_results.json
```

### 7. 產生保留圖片

Dataset 圖：

```bash
python -m safe_gta.visualize_dataset
```

產出：

```text
safe_gta/results/figure/dataset_safe_vs_dangerous.png
safe_gta/results/figure/dataset_actions.png
```

Guided repair evolution 圖：

```bash
python -m safe_gta.visualize_guided_evolution --start_t 25 --beta 2.0
```

產出：

```text
safe_gta/results/figure/guided_repair_evolution_route.png
```

MetaDrive top-down replay 圖：

```bash
python -m safe_gta.render_topdown_comparison
```

產出：

```text
safe_gta/results/final_figures/topdown_baseline.png
safe_gta/results/final_figures/topdown_repaired.png
safe_gta/results/final_figures/topdown_comparison.png
safe_gta/results/metrics/topdown_comparison_summary.json
safe_gta/results/artifacts/topdown_comparison_actions.npz
```

Poster 圖：

```bash
python generate_final_poster.py
```

產出：

```text
safe_gta/results/figure/safe_gta_final_poster.png
```

## 每支 code 的用途

### `download_dataset.py`

下載 MetaDrive offline safe RL dataset。這支在根目錄，不在 `safe_gta/` 裡。

### `evaluate_repair_3d.py`

固定同一張 MetaDrive 地圖，先跑 baseline risky action，再重播 repaired action。它主要是驗證 pipeline 骨架，方便未來把真正的 Safe-GTA repair 接進去。

產出：

```text
safe_gta/results/metrics/repair_3d_summary.json
safe_gta/results/artifacts/repair_3d_actions.npz
```

### `generate_final_poster.py`

讀取 replay action，畫出報告用 2D poster 圖。

### `safe_gta/train_diffusion.py`

訓練 diffusion trajectory model。它會讀 MetaDrive HDF5，把 observation/action 合成 trajectory，學習 trajectory prior。

### `safe_gta/safety_critic.py`

訓練 Safety Critic。輸入是 `(observation, action)`，輸出是 predicted safety cost。這個 critic 之後會在 guided denoising 裡提供 safety gradient。

### `safe_gta/generate_safe_gta.py`

Safe-GTA 的核心生成腳本。它會：

1. 載入 diffusion checkpoint。
2. 載入 Safety Critic checkpoint。
3. 從 MetaDrive dataset 抽 trajectory。
4. 用 SDEdit 做 repair。
5. 在 denoising 中加入 Safety Critic guidance。
6. 挑 predicted cost 低的 trajectory，存成 augmented dataset。

### `safe_gta/train_cql.py`

訓練 CQL。沒有給 `--augmented_data` 時就是 Baseline 1；給 `augmented_no_safety.pkl` 時是 Baseline 2；給 `augmented_safe_gta...pkl` 時是 Safe-GTA。

### `safe_gta/baseline2/generate_augmented.py`

Baseline 2 的資料生成器。它只使用 diffusion repair，不使用 Safety Critic guidance，並將 generated data 的 cost label 設成 0。

### `safe_gta/render_topdown_comparison.py`

用 MetaDrive 原生 top-down renderer 畫固定地圖 replay。這是目前最適合給 TA 看「baseline 會 out of road、repaired 可以延續且 cost=0」的圖。

目前結果：

```text
Baseline: steps=60, cost=1.0, out_of_road=True
Repaired: steps=130, cost=0.0, out_of_road=False
```

### `safe_gta/visualize_dataset.py`

產生 dataset safe/dangerous 對比圖和 action 圖，對應你保留在 `results/figure/` 的兩張 dataset 圖。

### `safe_gta/visualize_guided_evolution.py`

產生 Safe-GTA guided denoising 過程圖，對應 `guided_repair_evolution_route.png`。

### `safe_gta/diffusion/ddpm.py`

DDPM diffusion 流程本體，包含 sampling / repair 相關方法。

### `safe_gta/diffusion/noise_schedule.py`

Diffusion noise schedule。

### `safe_gta/diffusion/unet1d.py`

1D temporal U-Net，用來處理 trajectory sequence。

## 已刪掉的類型

已刪掉的主要是：

- 舊 Toy Task code。
- 舊 proxy / schematic 視覺化 code。
- 不會生成你保留圖片或影片的舊視覺化 code。
- 重複或舊版下載腳本。
- 環境檢查腳本。

## 報告時要誠實說明

- 目前 fixed-map replay 的 repair hook 還是 mock action repair，不是真正完整 Diffusion + Safety Critic 3D closed-loop repair。
- Safe-GTA offline 數字來自 dataset labels / learned critic，不等於 trained CQL policy 的真實 simulator rollout。
- Baseline 2 的 cost=0 是人工標籤設定，不代表真實安全。
