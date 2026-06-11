import argparse
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from aim import Run

from safetensors import safe_open

from model import Transformer, TransSmall, TransXSmall, TransXXSmall, TransSmall_rope, TransXSmall_rope, TransXXSmall_rope


class MazeDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        with safe_open(data_path, framework="pt") as f:
            s = f.get_slice("puzzle")
            shape = s.get_shape()
            self.n = shape[0]
            self.l = shape[1]
            self.g1 = shape[2]
            self.h = shape[3]
            self.w = shape[4]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        with safe_open(self.data_path, framework="pt") as f:
            puzzle = f.get_slice("puzzle")[idx].float()
            solution = f.get_slice("solution")[idx].float()
        puzzle = puzzle.reshape(self.l * self.g1, self.h, self.w)
        solution = 2.0 * solution - 1.0
        return puzzle, solution


def parse_ch_mults(s: str) -> tuple[int, ...]:
    return tuple(int(m) for m in s.split(","))


@torch.no_grad()
def evaluate(model, val_loader, device, num_steps=20, global_step=0, aim_run=None, amp=False):
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    val_loss = 0.0
    val_batches = 0
    solved = 0
    total_mazes = 0

    for puzzle, solution in val_loader:
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B, out_ch, H, W = solution.shape

        with torch.amp.autocast("cuda", enabled=amp):
            t = torch.rand(B, device=device)
            x_0 = torch.randn_like(solution)
            x_1 = solution
            x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
            v_target = x_1 - x_0
            model_input = torch.cat([puzzle, x_t], dim=1)
            v_pred = model(model_input, t)
            val_loss += F.mse_loss(v_pred, v_target).item()
        val_batches += 1

        x = torch.randn_like(solution)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_i = torch.full((B,), i * dt, device=device)
            with torch.amp.autocast("cuda", enabled=amp):
                v = model(torch.cat([puzzle, x], dim=1), t_i)
            x = x + dt * v
        pred = ((x + 1) / 2 > 0.5).float()
        gt = ((solution + 1) / 2 > 0.5).float()

        total_correct += (pred == gt).sum().item()
        total_pixels += gt.numel()
        total_intersection += (pred * gt).sum().item()
        total_union += ((pred + gt) > 0).sum().item()

        pred_flat = pred.view(B, -1)
        gt_flat = gt.view(B, -1)
        intersection = (pred_flat * gt_flat).sum(dim=1).float()
        union = ((pred_flat + gt_flat) > 0).sum(dim=1).float()
        all_match = (pred_flat == gt_flat).all(dim=1)
        perfect = all_match | ((intersection / union.clamp(min=1)) == 1.0)
        solved += perfect.sum().item()
        total_mazes += B

    avg_val_loss = val_loss / max(val_batches, 1)
    pixel_acc = total_correct / max(total_pixels, 1)
    iou = total_intersection / max(total_union, 1)
    solve_rate = solved / max(total_mazes, 1)

    if aim_run is not None:
        aim_run.track(avg_val_loss, name="loss", step=global_step, context={"subset": "val"})
        aim_run.track(pixel_acc, name="pixel_accuracy", step=global_step, context={"subset": "val"})
        aim_run.track(iou, name="iou", step=global_step, context={"subset": "val"})
        aim_run.track(solve_rate, name="solve_rate", step=global_step, context={"subset": "val"})

    model.train()
    return avg_val_loss, pixel_acc, iou, solve_rate


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Transformer maze solver training")
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--time_emb_dim", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aim_repo", type=str, default=".aim")
    parser.add_argument("--aim_experiment", type=str, default="DiffuMaze-FlowMatching-Trans")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision (default: on)")
    parser.add_argument("--model", type=str, default="trans_small", choices=["trans_small", "trans_xsmall", "trans_xxsmall", "trans_small_rope", "trans_xsmall_rope", "trans_xxsmall_rope"], help="Model variant to use")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Enable torch.compile (default: on)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing (default: on)")
    parser.add_argument("--load-prev", action="store_true", help="Resume from latest checkpoint in checkpoint_dir")
    args = parser.parse_args()

    model_arch_defaults = {
        "trans_small":       {"hidden_size": 288, "depth": 5, "num_heads": 4, "mlp_ratio": 3, "patch_size": 2, "time_emb_dim": 576},
        "trans_xsmall":      {"hidden_size": 128, "depth": 5, "num_heads": 4, "mlp_ratio": 2, "patch_size": 2, "time_emb_dim": 256},
        "trans_xxsmall":     {"hidden_size": 80, "depth": 3, "num_heads": 4, "mlp_ratio": 2, "patch_size": 2, "time_emb_dim": 80},
        "trans_small_rope":  {"hidden_size": 288, "depth": 5, "num_heads": 4, "mlp_ratio": 3, "patch_size": 2, "time_emb_dim": 576},
        "trans_xsmall_rope": {"hidden_size": 128, "depth": 5, "num_heads": 4, "mlp_ratio": 2, "patch_size": 2, "time_emb_dim": 256},
        "trans_xxsmall_rope":{"hidden_size": 80, "depth": 3, "num_heads": 4, "mlp_ratio": 2, "patch_size": 2, "time_emb_dim": 80},
    }
    for k, v in model_arch_defaults[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    start_epoch = 0
    load_ckpt = None

    if args.load_prev:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        runs = sorted([d for d in os.listdir(args.checkpoint_dir) if d.startswith("run_")])
        if not runs:
            raise ValueError(f"No previous runs found in {args.checkpoint_dir}")
        latest_run = runs[-1]
        run_dir = os.path.join(args.checkpoint_dir, latest_run)
        run_id = int(latest_run.split("_")[1])
        ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("epoch_") and f.endswith(".pt")])
        if not ckpts:
            raise ValueError(f"No checkpoints found in {run_dir}")
        latest_ckpt = os.path.join(run_dir, ckpts[-1])
        if rank == 0:
            print(f"Resuming from {latest_ckpt}")
        load_ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        start_epoch = load_ckpt["epoch"] + 1
        args.hidden_size = load_ckpt["hidden_size"]
        args.depth = load_ckpt["depth"]
        args.num_heads = load_ckpt["num_heads"]
        args.mlp_ratio = load_ckpt["mlp_ratio"]
        args.patch_size = load_ckpt["patch_size"]
        args.time_emb_dim = load_ckpt["time_emb_dim"]
        args.model = load_ckpt.get("model_name", "trans_small")
    else:
        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            existing = [d for d in os.listdir(args.checkpoint_dir) if d.startswith("run_")]
            run_id = max((int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()), default=-1) + 1
            run_dir = os.path.join(args.checkpoint_dir, f"run_{run_id:03d}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = None

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    full_dataset = MazeDataset(args.data_path)
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    puzzle_sample, solution_sample = full_dataset[0]
    in_channels = puzzle_sample.shape[0] + solution_sample.shape[0]
    out_channels = solution_sample.shape[0]

    model_class = {
        "trans_xsmall": TransXSmall,
        "trans_xxsmall": TransXXSmall,
        "trans_small_rope": TransSmall_rope,
        "trans_xsmall_rope": TransXSmall_rope,
        "trans_xxsmall_rope": TransXXSmall_rope,
    }.get(args.model, TransSmall)
    model = model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        time_emb_dim=args.time_emb_dim,
        use_checkpoint=args.checkpoint,
    ).to(device)

    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if load_ckpt is not None:
        scheduler.load_state_dict(load_ckpt["scheduler_state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if load_ckpt is not None:
        state_dict = load_ckpt["model_state_dict"]
        load_target = model.module if world_size > 1 else model

        sd_has_orig = any(k.startswith("_orig_mod.") for k in state_dict)
        target_has_orig = any(k.startswith("_orig_mod.") for k in load_target.state_dict())

        if sd_has_orig and not target_has_orig:
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        elif not sd_has_orig and target_has_orig:
            state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}

        load_target.load_state_dict(state_dict)
        optimizer.load_state_dict(load_ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if rank == 0:
        aim_run = Run(
            repo=args.aim_repo,
            experiment=args.aim_experiment,
        )
        aim_run["hparams"] = {
            "run_id": run_id,
            "run_dir": run_dir,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size * world_size,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "patch_size": args.patch_size,
            "time_emb_dim": args.time_emb_dim,
            "world_size": world_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "eval_steps": args.eval_steps,
            "amp": args.amp,
            "compile": args.compile,
            "model": args.model,
            "model_params": sum(p.numel() for p in model.parameters()),
            "load_prev": args.load_prev,
            "start_epoch": start_epoch,
        }
    else:
        aim_run = None

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for puzzle, solution in train_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            batch_size = puzzle.shape[0]

            t = torch.rand(batch_size, device=device)
            x_0 = torch.randn_like(solution)
            x_1 = solution
            x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
            v_target = x_1 - x_0

            with torch.amp.autocast("cuda", enabled=args.amp):
                model_input = torch.cat([puzzle, x_t], dim=1)
                v_pred = model(model_input, t)
                loss = F.mse_loss(v_pred, v_target) / args.grad_accum_steps

            scaler.scale(loss).backward()

            num_batches += 1
            epoch_loss += loss.item() * args.grad_accum_steps

            if num_batches % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if rank == 0 and num_batches % args.log_every == 0:
                global_step = epoch * len(train_loader) + num_batches
                aim_run.track(loss.item() * args.grad_accum_steps, name="loss", step=global_step, context={"subset": "train"})
                aim_run.track(scheduler.get_last_lr()[0], name="lr", step=global_step, context={"subset": "train"})

        if num_batches % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if rank == 0:
            print(f"Epoch {epoch}: train_loss={avg_loss:.6f}", end="")

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                global_step = (epoch + 1) * len(train_loader)
                avg_val_loss, pixel_acc, iou, solve_rate = evaluate(
                    model, val_loader, device,
                    num_steps=args.eval_steps,
                    global_step=global_step,
                    aim_run=aim_run,
                    amp=args.amp,
                )
                print(f" | val_loss={avg_val_loss:.6f} pixel_acc={pixel_acc:.4f} iou={iou:.4f} solve_rate={solve_rate:.4f}", end="")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": avg_loss,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "hidden_size": args.hidden_size,
                        "depth": args.depth,
                        "num_heads": args.num_heads,
                        "mlp_ratio": args.mlp_ratio,
                        "patch_size": args.patch_size,
                        "time_emb_dim": args.time_emb_dim,
                        "model_name": args.model,
                    },
                    os.path.join(run_dir, f"epoch_{epoch:04d}.pt"),
                )

            print()

    if rank == 0:
        aim_run.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()