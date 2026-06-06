# Safe-GTA Final Project

Safe-GTA is a MetaDrive offline RL trajectory-repair project. The project goal is to use a diffusion trajectory model plus a learned Safety Critic to generate safer augmented trajectories for CQL training.

Current status: the core offline Safe-GTA pipeline is implemented and runnable. The current evidence is based on offline dataset labels and learned Safety Critic scores, not a full MetaDrive simulator rollout evaluation.

## Pipeline

```text
MetaDrive offline HDF5 dataset
        |
        |-- train_diffusion.py
        |       -> diffusion trajectory prior
        |
        |-- safety_critic.py
        |       -> learned per-step safety cost model
        |
        |-- generate_safe_gta.py
        |       -> Safety-Critic-guided augmented trajectories
        |
        |-- train_cql.py
                -> CQL policy trained on raw / GTA / Safe-GTA data
```

## Main Commands

Activate the environment from the project root:

```bash
conda activate safe_gta
cd "C:\Users\kenny\Desktop\School\nycu course\RL\Lab\final project"
```

Train diffusion and Safety Critic:

```bash
python -m safe_gta.train_diffusion --epochs 200 --batch_size 64
python -m safe_gta.safety_critic --epochs 20 --max_samples 200000
```

Generate vanilla GTA data and train Baseline 2:

```bash
python -m safe_gta.baseline2.generate_augmented --checkpoint safe_gta/checkpoints/diffusion_metadrive_final.pt --n_augmented 5000 --start_t 150 --output safe_gta/data/augmented_no_safety.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl --epochs 50
```

Generate Safe-GTA data and train CQL:

```bash
python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 2.0 --start_t 25 --output safe_gta/data/augmented_safe_gta_t25_beta20.pkl
python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_safe_gta_t25_beta20.pkl --epochs 50
```

Generate figures:

```bash
python -m safe_gta.visualize_results
python -m safe_gta.visualize_guided_evolution --start_t 25 --beta 2.0 --search_n 2000 --unsafe_min_cost 1.0
python -m safe_gta.visualize_topdown_repair --start_t 25 --beta 2.0 --search_n 2000 --unsafe_min_cost 1.0
python -m safe_gta.visualize_turn_repair --index 12841 --start_t 150 --beta 10.0 --seed 17 --output safe_gta/results/metadrive_turn_large_repair.png
```

3D validation skeleton and poster figure:

```bash
python evaluate_repair_3d.py --no-render --steps 80 --pause-before-replay 0 --bad-policy late_left --throttle 0.65 --steering-scale 0.45 --warmup-steps 40 --repair-mode straighten
python evaluate_repair_3d.py --render --seed 42 --steps 80 --bad-policy late_left --throttle 0.65 --steering-scale 0.45 --warmup-steps 40 --repair-mode straighten
python generate_final_poster.py --extend-good 100 --extend-mode straight
```

## Current Results

Safety Critic validation:

```text
accuracy = 0.991975
safe mean predicted cost = 0.0065
dangerous mean predicted cost = 0.9812
```

Safe-GTA generation ablation:

```text
start_t=150, beta=0.1 -> mean predicted cost 0.9574, pass rate 0.0%
start_t=150, beta=1.0 -> mean predicted cost 0.9524, pass rate 0.0%
start_t=150, beta=2.0 -> mean predicted cost 0.9457, pass rate 0.0%
start_t=25,  beta=2.0 -> mean predicted cost 0.0891, pass rate 100.0%
```

CQL offline label-based comparison:

```text
Baseline 1 Raw CQL: normalized_return 0.3806, cost 5.4962
Baseline 2 CQL + GTA: normalized_return 1.0, cost 0.0
Safe-GTA CQL: normalized_return 1.0, cumulative predicted cost 17.8172
```

Important interpretation: Baseline 2 has artificial zero-cost labels in the generated data. Safe-GTA uses learned critic costs, and the CQL cost is cumulative. These numbers should not be interpreted as real simulator safety performance.

## Key Outputs

Small result figures and JSON files are kept in Git. Large datasets, augmented `.pkl` files, HDF5 files, `.npz` action dumps, and model checkpoints are ignored.

Useful generated figures:

```text
safe_gta/results/baseline_comparison.png
safe_gta/results/diffusion_metadrive_loss.png
safe_gta/results/metadrive_safety_critic_costs.png
safe_gta/results/guided_repair_evolution_route.png
safe_gta/results/guided_repair_topdown_route.png
safe_gta/results/metadrive_turn_large_repair.png
safe_gta/results/safe_gta_final_poster.png
```

## Limitations

What is complete:

- MetaDrive offline dataset loading
- diffusion trajectory model
- Safety Critic training
- Safety-Critic-guided denoising
- Safe-GTA augmented dataset generation
- CQL training on raw / GTA / Safe-GTA data
- qualitative visualization and 3D validation skeleton

What is still future work:

- real MetaDrive simulator rollout of the trained CQL policy
- FISOR baseline comparison
- formal trajectory fidelity metric such as Frechet distance
- broader `start_t` / `beta` ablation

## Report

The detailed experiment notes are in:

```text
報告.md
```
