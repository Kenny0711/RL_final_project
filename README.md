# Safe-GTA Final Project

這是 RL final project 的 Safe-GTA 專案整理版。主軸是用 **Diffusion Model** 修復 MetaDrive offline trajectory，再用 **Safety Critic** 在 denoising 過程做安全引導，最後產生 augmented dataset 給 CQL 訓練。

## 專案目前能做什麼

1. 下載 MetaDrive offline HDF5 dataset。
2. 訓練 diffusion trajectory model。
3. 訓練 Safety Critic。
4. 用 CQL 訓練 raw / GTA / Safe-GTA 資料。
5. 產生報告用圖片與 fixed-map top-down replay 對比圖。

## 快速執行順序

```bash
conda activate safe_gta
cd "C:\Users\kenny\Desktop\School\nycu course\RL\Lab\final project"
```

下載資料：

```bash
python download_dataset.py
```

訓練 diffusion 與 Safety Critic：

```bash
python -m safe_gta.train_diffusion --epochs 200 --batch_size 64
python -m safe_gta.safety_critic --epochs 20 --max_samples 200000
```

Baseline 1：

```bash
python -m safe_gta.train_cql --epochs 50
```

Baseline 2：

```bash
python -m safe_gta.baseline2.generate_augmented --checkpoint safe_gta/checkpoints/diffusion_metadrive_final.pt --n_augmented 5000 --start_t 150 --output safe_gta/data/augmented_no_safety.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl --epochs 50
```

Safe-GTA：

```bash
python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 2.0 --start_t 25 --output safe_gta/data/augmented_safe_gta_t25_beta20.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_safe_gta_t25_beta20.pkl --epochs 50
```

產生保留的結果圖：

```bash
python -m safe_gta.visualize_dataset
python -m safe_gta.visualize_guided_evolution --start_t 25 --beta 2.0
python -m safe_gta.render_topdown_comparison
python generate_final_poster.py
```

## 重要結果資料夾

```text
safe_gta/results/figure/
```

放報告用圖片與影片，例如 dataset 圖、guided repair 圖、poster 圖、MetaDrive 影片。

```text
safe_gta/results/final_figures/
```

放 fixed-map MetaDrive top-down replay 圖：

```text
topdown_baseline.png
topdown_repaired.png
topdown_comparison.png
```

```text
safe_gta/results/metrics/
```

放 JSON 實驗數值。

```text
safe_gta/results/artifacts/
```

放 replay actions 等暫存檔。

## 每個 code 在做什麼

### 根目錄

```text
download_dataset.py
```

下載 MetaDrive offline HDF5 dataset 到 `safe_gta/data/`。

```text
evaluate_repair_3d.py
```

固定 MetaDrive seed，先跑 baseline risky action，再跑 repaired action。這是 3D / replay 驗證骨架，目前 repair hook 還是 mock，之後可接真正 Diffusion + Safety Critic。

```text
generate_final_poster.py
```

用 replay action 產生 2D poster 圖，輸出到 `safe_gta/results/figure/safe_gta_final_poster.png`。

### `safe_gta/`

```text
train_diffusion.py
```

訓練 diffusion trajectory model，學 MetaDrive trajectory prior。

```text
safety_critic.py
```

訓練 Safety Critic，預測 `(observation, action)` 的 safety cost。

```text
generate_safe_gta.py
```

Safe-GTA 核心生成腳本。用 diffusion repair + Safety Critic guidance 產生安全引導後的 augmented dataset。

```text
train_cql.py
```

訓練 CQL。可用 raw dataset、Baseline 2 augmented dataset、Safe-GTA augmented dataset。

```text
baseline2/generate_augmented.py
```

Baseline 2。只用 diffusion / GTA 產生 augmented data，不使用 Safety Critic guidance。

```text
render_topdown_comparison.py
```

用 MetaDrive 原生 top-down renderer，在固定 seed=42 的同一張地圖上比較 baseline 和 repaired action。

```text
visualize_dataset.py
```

產生保留的 dataset 圖：

```text
safe_gta/results/figure/dataset_safe_vs_dangerous.png
safe_gta/results/figure/dataset_actions.png
```

```text
visualize_guided_evolution.py
```

產生 guided repair evolution 圖：

```text
safe_gta/results/figure/guided_repair_evolution_route.png
```

### `safe_gta/diffusion/`

```text
unet1d.py
noise_schedule.py
ddpm.py
```

Diffusion model 的核心模組。`train_diffusion.py`、`baseline2/generate_augmented.py` 和 `generate_safe_gta.py` 都會用到。

