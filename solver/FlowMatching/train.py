import argparse
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from aim import Run

from safetensors.torch import load_file

from model import UNet


class MazeDataset(Dataset):
    def __init__(self, data_path: str):
        data = load_file(data_path)
        self.puzzle = data["puzzle"].float()
        self.solution = data["solution"].float()

    def __len__(self) -> int:
        return len(self.puzzle)

    def __getitem__(self, idx: int):
        puzzle = self.puzzle[idx]
        solution = self.solution[idx]
        l, g1, h, w = puzzle.shape
        puzzle = puzzle.reshape(l * g1, h, w)
        solution = 2.0 * solution - 1.0
        return puzzle, solution


def parse_ch_mults(s: str) -> tuple[int, ...]:
    return tuple(int(m) for m in s.split(","))


@torch.no_grad()
def evaluate(model, val_loader, device, num_steps=20, global_step=0, aim_run=None):
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    val_loss = 0.0
    val_batches = 0

    for puzzle, solution in val_loader:
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B, out_ch, H, W = solution.shape

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
            v = model(torch.cat([puzzle, x], dim=1), t_i)
            x = x + dt * v
        pred = ((x + 1) / 2 > 0.5).float()
        gt = ((solution + 1) / 2 > 0.5).float()

        total_correct += (pred == gt).sum().item()
        total_pixels += gt.numel()
        total_intersection += (pred * gt).sum().item()
        total_union += ((pred + gt) > 0).sum().item()

    avg_val_loss = val_loss / max(val_batches, 1)
    pixel_acc = total_correct / max(total_pixels, 1)
    iou = total_intersection / max(total_union, 1)

    if aim_run is not None:
        aim_run.track(avg_val_loss, name="loss", step=global_step, context={"subset": "val"})
        aim_run.track(pixel_acc, name="pixel_accuracy", step=global_step, context={"subset": "val"})
        aim_run.track(iou, name="iou", step=global_step, context={"subset": "val"})

    model.train()
    return avg_val_loss, pixel_acc, iou


def main():
    parser = argparse.ArgumentParser(description="Flow Matching maze solver training")
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--ch_mults", type=str, default="1,2,4,8")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--time_emb_dim", type=int, default=256)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aim_repo", type=str, default=".aim")
    parser.add_argument("--aim_experiment", type=str, default="DiffuMaze-FlowMatching")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

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
    ch_mults = parse_ch_mults(args.ch_mults)

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_ch=args.base_ch,
        ch_mults=ch_mults,
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size * world_size,
            "base_ch": args.base_ch,
            "ch_mults": list(ch_mults),
            "num_res_blocks": args.num_res_blocks,
            "time_emb_dim": args.time_emb_dim,
            "world_size": world_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "eval_steps": args.eval_steps,
        }
    else:
        aim_run = None

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for puzzle, solution in train_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            batch_size = puzzle.shape[0]

            t = torch.rand(batch_size, device=device)
            x_0 = torch.randn_like(solution)
            x_1 = solution
            x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
            v_target = x_1 - x_0

            model_input = torch.cat([puzzle, x_t], dim=1)
            v_pred = model(model_input, t)

            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if rank == 0 and num_batches % args.log_every == 0:
                global_step = epoch * len(train_loader) + num_batches
                aim_run.track(loss.item(), name="loss", step=global_step, context={"subset": "train"})
                aim_run.track(scheduler.get_last_lr()[0], name="lr", step=global_step, context={"subset": "train"})

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if rank == 0:
            print(f"Epoch {epoch}: train_loss={avg_loss:.6f}", end="")

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                global_step = (epoch + 1) * len(train_loader)
                avg_val_loss, pixel_acc, iou = evaluate(
                    model, val_loader, device,
                    num_steps=args.eval_steps,
                    global_step=global_step,
                    aim_run=aim_run,
                )
                print(f" | val_loss={avg_val_loss:.6f} pixel_acc={pixel_acc:.4f} iou={iou:.4f}", end="")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "base_ch": args.base_ch,
                        "ch_mults": list(ch_mults),
                        "num_res_blocks": args.num_res_blocks,
                        "time_emb_dim": args.time_emb_dim,
                    },
                    os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt"),
                )

            print()

    if rank == 0:
        aim_run.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()