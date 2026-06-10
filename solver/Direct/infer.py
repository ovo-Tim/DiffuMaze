import argparse

import torch
from safetensors.torch import load_file, save_file

from model import DirectUNet, DirectUNetSmall, DirectUNetXSmall, DirectUNetXXSmall


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_class = {
        "unet_small": DirectUNetSmall, "unet_xsmall": DirectUNetXSmall, "unet_xxsmall": DirectUNetXXSmall,
    }.get(checkpoint.get("model_name", "unet"), DirectUNet)

    model = model_class(
        in_channels=checkpoint["in_channels"],
        out_channels=checkpoint["out_channels"],
        base_ch=checkpoint["base_ch"],
        ch_mults=tuple(checkpoint["ch_mults"]),
        num_res_blocks=checkpoint["num_res_blocks"],
    )
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, puzzle, device="cuda"):
    pred = model(puzzle)
    solution = (pred + 1) / 2
    return (solution > 0.5).float()


def main():
    parser = argparse.ArgumentParser(description="Direct maze solver inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--output_path", type=str, default="solution.safetensors")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)

    data = load_file(args.data_path)
    puzzle = data["puzzle"].float()
    n, l, g1, h, w = puzzle.shape
    puzzle = puzzle.reshape(n, l * g1, h, w)

    all_solutions = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = puzzle[start:end].to(args.device)
        sol = predict(model, batch, args.device)
        all_solutions.append(sol.cpu())

    solution = torch.cat(all_solutions, dim=0)
    solution_int8 = solution.to(torch.int8)

    save_file({"solution": solution_int8}, args.output_path)
    print(f"Saved solution to {args.output_path}, shape: {solution_int8.shape}")


if __name__ == "__main__":
    main()