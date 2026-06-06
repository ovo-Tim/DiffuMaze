# Diffusion-based maze solver

Flow Matching model for solving multi-layer multi-goal mazes.
Uses a UNet to learn the velocity field from noise to solution, conditioned on the maze puzzle.

## Setup

```bash
cd solver
uv sync
```

## Training

All training scripts are under `solver/FlowMatching/`.

### Single GPU

```bash
cd solver/FlowMatching
python train.py --data_path ../maze.safetensors --batch_size 256
```

### Multi-GPU DDP (7 GPUs)

```bash
cd solver/FlowMatching
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 train.py \
    --data_path ../maze.safetensors \
    --batch_size 256 \
    --epochs 200
```

`--amp` and `--compile` are **on by default**. Disable with `--no-amp` or `--no-compile` (e.g. if you run out of VRAM during compilation).

When increasing batch size, scale LR correspondingly (e.g. batch 64→256, LR 1e-4→4e-4).

Recommended fast config:
```bash
torchrun --nproc_per_node=7 train.py \
    --data_path ../maze.safetensors \
    --batch_size 256 --lr 4e-4
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `maze.safetensors` | Path to maze data |
| `--batch_size` | 64 | Per-GPU batch size |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--epochs` | 200 | Number of epochs |
| `--base_ch` | 64 | UNet base channels |
| `--ch_mults` | `1,2,4,8` | Channel multipliers per level |
| `--num_res_blocks` | 2 | ResBlocks per level |
| `--time_emb_dim` | 256 | Time embedding dimension |
| `--checkpoint_dir` | `checkpoints` | Checkpoint save directory |
| `--save_every` | 10 | Save checkpoint every N epochs |
| `--log_every` | 50 | Log training loss every N batches |
| `--eval_every` | 5 | Run validation every N epochs |
| `--val_ratio` | 0.1 | Fraction of data held out for validation |
| `--eval_steps` | 20 | Euler steps used during validation |
| `--seed` | 42 | Random seed |
| `--aim_repo` | `.aim` | Aim repo directory |
| `--aim_experiment` | `DiffuMaze-FlowMatching` | Aim experiment name |
| `--amp` | on | Mixed precision training. Use `--no-amp` to disable |
| `--compile` | on | `torch.compile` on the UNet. Use `--no-compile` to disable |

## Monitoring

Training metrics are tracked with [Aim](https://github.com/aimhubio/aim).

```bash
cd solver/FlowMatching
aim up
```

Opens a local web UI at `http://127.0.0.1:43800`.

Tracked metrics:
- **Train/val loss** – MSE velocity loss
- **Pixel accuracy** – Binary solution path accuracy
- **IoU** – Intersection over Union for solution paths
- **Learning rate** – Cosine schedule

## Inference

```bash
cd solver/FlowMatching
python infer.py \
    --checkpoint checkpoints/epoch_0199.pt \
    --data_path ../maze.safetensors \
    --output_path solution.safetensors \
    --method euler \
    --num_steps 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint |
| `--data_path` | `maze.safetensors` | Input puzzle data |
| `--output_path` | `solution.safetensors` | Output solution file |
| `--num_steps` | 100 | Euler/RK4 integration steps |
| `--method` | `euler` | Sampler: `euler` or `rk4` |
| `--batch_size` | 16 | Batch size for batched inference |
| `--device` | `cuda` | Device to run on |

## Data Format

Uses the same safetensors schema as the Rust generator in `generator/`.

- **`puzzle`**: shape `(n, l, g+1, h, w)`, int8 — walls (ch 0) + route checkpoints (ch 1..g)
- **`solution`**: shape `(n, l, h, w)`, int8 — binary solution paths per layer