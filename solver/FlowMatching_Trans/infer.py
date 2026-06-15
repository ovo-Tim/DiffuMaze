import argparse

import torch
from safetensors.torch import load_file, save_file

from model import TransSmall, TransSmall_rope, TransXSmall, TransXSmall_rope, TransXXSmall, TransXXSmall_rope


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_class = {
        "trans_small": TransSmall,
        "trans_xsmall": TransXSmall,
        "trans_xxsmall": TransXXSmall,
        "trans_small_rope": TransSmall_rope,
        "trans_xsmall_rope": TransXSmall_rope,
        "trans_xxsmall_rope": TransXXSmall_rope,
    }.get(checkpoint.get("model_name", "trans_small"), TransSmall)
    model = model_class(
        in_channels=checkpoint["in_channels"],
        out_channels=checkpoint["out_channels"],
        hidden_size=checkpoint["hidden_size"],
        depth=checkpoint["depth"],
        num_heads=checkpoint["num_heads"],
        mlp_ratio=checkpoint["mlp_ratio"],
        patch_size=checkpoint["patch_size"],
        time_emb_dim=checkpoint["time_emb_dim"],
    )
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def euler_sample(model, puzzle, num_steps=100, device="cuda"):
    out_channels = model.out_channels
    B, _, H, W = puzzle.shape
    x = torch.randn(B, out_channels, H, W, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        model_input = torch.cat([puzzle, x], dim=1)
        v = model(model_input, t)
        x = x + dt * v

    solution = (x + 1) / 2
    return (solution > 0.5).float()


@torch.no_grad()
def rk4_sample(model, puzzle, num_steps=100, device="cuda"):
    out_channels = model.out_channels
    B, _, H, W = puzzle.shape
    x = torch.randn(B, out_channels, H, W, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_i = i * dt

        t = torch.full((B,), t_i, device=device)
        k1 = model(torch.cat([puzzle, x], dim=1), t)

        t_mid = torch.full((B,), t_i + 0.5 * dt, device=device)
        k2 = model(torch.cat([puzzle, x + 0.5 * dt * k1], dim=1), t_mid)

        k3 = model(torch.cat([puzzle, x + 0.5 * dt * k2], dim=1), t_mid)

        t_end = torch.full((B,), t_i + dt, device=device)
        k4 = model(torch.cat([puzzle, x + dt * k3], dim=1), t_end)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    solution = (x + 1) / 2
    return (solution > 0.5).float()


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Transformer maze solver inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--output_path", type=str, default="solution.safetensors")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--method", type=str, choices=["euler", "rk4"], default="euler")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)

    data = load_file(args.data_path)
    puzzle = data["puzzle"].float()
    n, l, g1, h, w = puzzle.shape
    puzzle = puzzle.reshape(n, l * g1, h, w)

    sample_fn = euler_sample if args.method == "euler" else rk4_sample

    all_solutions = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = puzzle[start:end].to(args.device)
        sol = sample_fn(model, batch, args.num_steps, args.device)
        all_solutions.append(sol.cpu())

    solution = torch.cat(all_solutions, dim=0)
    solution_int8 = solution.to(torch.int8)

    save_file({"solution": solution_int8}, args.output_path)
    print(f"Saved solution to {args.output_path}, shape: {solution_int8.shape}")


if __name__ == "__main__":
    main()
