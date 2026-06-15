import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors import safe_open
from model import TransSmall, TransSmall_rope, TransXSmall, TransXSmall_rope, TransXXSmall, TransXXSmall_rope


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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Flow Matching Transformer model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of mazes to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    model_class = {
        "trans_small": TransSmall,
        "trans_xsmall": TransXSmall,
        "trans_xxsmall": TransXXSmall,
        "trans_small_rope": TransSmall_rope,
        "trans_xsmall_rope": TransXSmall_rope,
        "trans_xxsmall_rope": TransXXSmall_rope,
    }.get(ckpt.get("model_name", "trans_small"), TransSmall)

    model = model_class(
        in_channels=ckpt["in_channels"],
        out_channels=ckpt["out_channels"],
        hidden_size=ckpt["hidden_size"],
        depth=ckpt["depth"],
        num_heads=ckpt["num_heads"],
        mlp_ratio=ckpt["mlp_ratio"],
        patch_size=ckpt["patch_size"],
        time_emb_dim=ckpt["time_emb_dim"],
    ).to(device)

    sd_has_orig = any(k.startswith("_orig_mod.") for k in sd)
    if sd_has_orig:
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    dataset = MazeDataset(args.data_path)
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.num_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    val_loss = 0.0
    val_batches = 0
    solved = 0
    total_mazes = 0

    for puzzle, solution in loader:
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B = puzzle.shape[0]

        with torch.amp.autocast("cuda", enabled=args.amp):
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
        dt = 1.0 / args.num_steps
        for i in range(args.num_steps):
            t_i = torch.full((B,), i * dt, device=device)
            with torch.amp.autocast("cuda", enabled=args.amp):
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

    avg_loss = val_loss / max(val_batches, 1)
    pixel_acc = total_correct / max(total_pixels, 1)
    iou = total_intersection / max(total_union, 1)
    solve_rate = solved / max(total_mazes, 1)

    model_name = ckpt.get("model_name", "trans_small")
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Model:        {model_name}")
    print(f"Parameters:   {n_params:,}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Data:         {args.data_path} ({total_mazes} mazes)")
    print(f"Eval steps:   {args.num_steps}")
    print(f"Device:       {device}")
    print(f"Loss:         {avg_loss:.6f}")
    print(f"Pixel acc:    {pixel_acc:.4f}")
    print(f"IoU:          {iou:.4f}")
    print(f"Solve rate:   {solve_rate:.4f} ({solved}/{total_mazes})")


if __name__ == "__main__":
    main()
