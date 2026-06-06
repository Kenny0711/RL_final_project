# Safe-GTA

Safe-GTA is an offline RL data-repair pipeline for MetaDrive. The current codebase supports:

1. loading MetaDrive offline HDF5 datasets,
2. training a diffusion trajectory repair model,
3. training a MetaDrive Safety Critic from dataset costs,
4. generating vanilla GTA augmented data,
5. generating Safety-Critic-guided Safe-GTA augmented data,
6. training CQL on raw / GTA / Safe-GTA data,
7. plotting comparison figures.

Important: the current evaluation is still offline dataset evaluation. It is not yet a real MetaDrive simulator rollout.

## Environment

Use the existing Conda environment:

```bash
conda activate safe_gta
cd "C:\Users\kenny\Desktop\School\nycu course\RL\Lab\final project"
```

Expected key packages:

- PyTorch 2.6.0+cu124
- numpy 1.26.4
- h5py
- matplotlib

## 1. Download MetaDrive Offline Dataset

```bash
python download_dataset.py
```

Outputs:

```text
safe_gta/data/metadrive_mediumsparse.hdf5
safe_gta/data/metadrive_mediummean.hdf5
safe_gta/data/metadrive_mediumdense.hdf5
```

The code prefers the datasets in this order:

```text
metadrive_mediumsparse.hdf5 -> metadrive_mediummean.hdf5 -> metadrive_mediumdense.hdf5
```

## 2. Inspect Dataset

```bash
python -m safe_gta.visualize_dataset
```

Outputs:

```text
safe_gta/results/dataset_safe_vs_dangerous.png
safe_gta/results/dataset_actions.png
safe_gta/results/dataset_distribution.png
```

These plots show safe vs dangerous trajectories, steering/throttle behavior, and cost/reward distributions.

## 3. Train MetaDrive Diffusion Model

```bash
python -m safe_gta.train_diffusion --epochs 200 --batch_size 64
```

For a quick CPU smoke test:

```bash
python -m safe_gta.train_diffusion --epochs 1 --batch_size 8192 --base_ch 8 --device cpu
```

Outputs:

```text
safe_gta/checkpoints/diffusion_metadrive_epoch50.pt
safe_gta/checkpoints/diffusion_metadrive_epoch100.pt
safe_gta/checkpoints/diffusion_metadrive_epoch150.pt
safe_gta/checkpoints/diffusion_metadrive_epoch200.pt
safe_gta/checkpoints/diffusion_metadrive_final.pt
safe_gta/results/diffusion_metadrive_loss.png
```

The checkpoint stores normalization statistics. The Safety Critic and guided repair use the same normalized feature space.

## 4. Train MetaDrive Safety Critic

```bash
python -m safe_gta.safety_critic --epochs 20 --max_samples 200000
```

For a quick CPU smoke test:

```bash
python -m safe_gta.safety_critic --epochs 1 --max_samples 2000 --hidden 64 --device cpu --output safe_gta/checkpoints/_debug_safety_critic.pt
```

Outputs:

```text
safe_gta/checkpoints/metadrive_safety_critic.pt
safe_gta/results/metadrive_safety_critic_results.json
safe_gta/results/metadrive_safety_critic_costs.png
```

The critic predicts per-step cost from normalized `(observation, action)` features. Its gradient is used inside:

```text
DDPM.guided_repair()
```

## 5. Baseline 1: CQL On Raw Dataset

```bash
python -m safe_gta.train_cql --epochs 50
```

Outputs:

```text
safe_gta/checkpoints/baseline1_cql.pt
safe_gta/results/baseline1_cql_results.json
```

This is the raw MetaDrive offline dataset baseline.

## 6. Baseline 2: Vanilla GTA, No Safety Guidance

Generate augmented data:

```bash
python -m safe_gta.baseline2.generate_augmented --n_augmented 5000 --start_t 150
```

Train CQL on the augmented data:

```bash
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl --epochs 50
```

Outputs:

```text
safe_gta/data/augmented_no_safety.pkl
safe_gta/checkpoints/baseline2_cql.pt
safe_gta/results/baseline2_cql_results.json
```

This version uses diffusion repair but does not use Safety Critic guidance.

## 7. Safe-GTA: GTA With Safety Critic Guidance

Generate Safety-Critic-guided augmented data:

```bash
python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 0.5 --start_t 150
```

Train CQL on Safe-GTA data:

```bash
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_safe_gta.pkl --epochs 50
```

Outputs:

```text
safe_gta/data/augmented_safe_gta.pkl
safe_gta/checkpoints/safe_gta_cql.pt
safe_gta/results/safe_gta_cql_results.json
```

What `generate_safe_gta` does:

1. loads `diffusion_metadrive_final.pt`,
2. loads `metadrive_safety_critic.pt`,
3. samples MetaDrive trajectories,
4. runs SDEdit repair,
5. applies Safety Critic gradient guidance during denoising,
6. keeps the lowest predicted-cost trajectories,
7. saves the final augmented dataset.

Main knobs:

```text
--beta              strength of Safety Critic guidance
--start_t           SDEdit noise level; larger means stronger repair
--oversample        generate extra candidates, then keep lowest-cost ones
--accept_threshold  reporting threshold for selected trajectory cost
```

## 8. Visualize Repair And Final Comparison

Visualize before/after diffusion repair:

```bash
python -m safe_gta.visualize_repair --n 5 --start_t 150
```

Plot method comparison:

```bash
python -m safe_gta.visualize_results
```

Outputs:

```text
safe_gta/results/metadrive_repair_comparison.png
safe_gta/results/baseline_comparison.png
safe_gta/results/diffusion_metadrive_loss.png
```

`visualize_results` reads:

```text
safe_gta/results/baseline1_cql_results.json
safe_gta/results/baseline2_cql_results.json
safe_gta/results/safe_gta_cql_results.json
```

If `safe_gta_cql_results.json` is missing, the Safe-GTA bar is shown as pending.

## Recommended Full Run Order

```bash
conda activate safe_gta
cd "C:\Users\kenny\Desktop\School\nycu course\RL\Lab\final project"

python download_dataset.py
python -m safe_gta.visualize_dataset

python -m safe_gta.train_diffusion --epochs 200 --batch_size 64
python -m safe_gta.safety_critic --epochs 20 --max_samples 200000

python -m safe_gta.train_cql --epochs 50

python -m safe_gta.baseline2.generate_augmented --n_augmented 5000 --start_t 150
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl --epochs 50

python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 0.5 --start_t 150
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_safe_gta.pkl --epochs 50

python -m safe_gta.visualize_repair --n 5 --start_t 150
python -m safe_gta.visualize_results
```

## Current Project Status

Implemented:

- MetaDrive HDF5 loading
- diffusion trajectory repair
- vanilla GTA augmentation
- MetaDrive Safety Critic training
- Safety-Critic-guided diffusion repair
- Safe-GTA augmented dataset generation
- CQL training for raw / GTA / Safe-GTA datasets
- comparison plotting

Still not implemented:

- real MetaDrive simulator rollout evaluation
- FISOR baseline
- formal Frechet trajectory fidelity metric
- beta ablation table
- trajectory ghosting / cost heatmap for the final report
