import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import load_model
from utils.viz import render_colored, render_heatmap, concat_images


@torch.no_grad()
def step_by_step(model, puzzle, num_steps=100, record_every=5, device="cuda"):
    out_channels = model.out_channels
    B, _, H, W = puzzle.shape
    x = torch.randn(B, out_channels, H, W, device=device)
    dt = 1.0 / num_steps

    steps = []
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(torch.cat([puzzle, x], dim=1), t)
        x = x + dt * v
        if (i + 1) % record_every == 0 or i == num_steps - 1:
            steps.append(x.clone())

    return steps


def main():
    parser = argparse.ArgumentParser(description="Visualize denoising step by step")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="denoise_viz")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--gif", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=int, default=4)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)

    data = load_file(args.data_path)
    puzzle = data["puzzle"].float()
    solution = data["solution"].float()

    n, l, g1, h, w = puzzle.shape
    puzzle_flat = puzzle.reshape(n, l * g1, h, w)

    puzzle_sample = puzzle[args.sample_idx]
    sol_gt = solution[args.sample_idx]

    steps = step_by_step(
        model,
        puzzle_flat[args.sample_idx : args.sample_idx + 1].to(args.device),
        num_steps=args.num_steps,
        record_every=args.record_every,
        device=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Rendering {len(steps)} steps...")
    for step_idx, x_t in enumerate(steps):
        x_t = x_t[0].cpu()
        for layer_idx in range(l):
            p_chk = puzzle_sample[layer_idx]
            img = render_heatmap(p_chk, x_t[layer_idx], scale=args.scale)
            fname = os.path.join(
                args.output_dir, f"step_{step_idx:03d}_layer_{layer_idx}.png"
            )
            img.save(fname)

    print("Rendering final prediction vs ground truth...")
    x_final = steps[-1][0].cpu()
    sol_pred = (x_final + 1) / 2
    for layer_idx in range(l):
        p_chk = puzzle_sample[layer_idx]
        img = render_colored(p_chk, sol_pred[layer_idx], scale=args.scale)
        img.save(os.path.join(args.output_dir, f"pred_layer_{layer_idx}.png"))
        img = render_colored(p_chk, sol_gt[layer_idx], scale=args.scale)
        img.save(os.path.join(args.output_dir, f"gt_layer_{layer_idx}.png"))

    if args.gif:
        try:
            import imageio

            for layer_idx in range(l):
                frames = []
                for step_idx in range(len(steps)):
                    fn = os.path.join(
                        args.output_dir, f"step_{step_idx:03d}_layer_{layer_idx}.png"
                    )
                    if os.path.exists(fn):
                        frames.append(imageio.v2.imread(fn))
                if frames:
                    gif_path = os.path.join(
                        args.output_dir, f"denoise_layer_{layer_idx}.gif"
                    )
                    imageio.mimsave(gif_path, frames, fps=args.gif_fps)
                    print(f"Saved {gif_path}")
        except ImportError:
            print("Install imageio for GIFs: uv pip install imageio")

    print(f"Output saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
