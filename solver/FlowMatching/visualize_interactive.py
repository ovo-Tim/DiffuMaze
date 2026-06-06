import argparse
import time

import gradio as gr
import torch
from safetensors.torch import load_file

from infer import load_model
from viz import render_colored, render_heatmap, concat_images

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHED_HEATMAP = []
_CACHED_COLORED = []
_CACHED_METRICS = {}
_GT_IMG = None
_P_CHK = None


@torch.no_grad()
def run_denoise(sample_idx, layer_name, num_steps):
    global _MODEL, _PUZZLE, _SOLUTION, _N, _L, _G1, _H, _W
    global _CACHED_HEATMAP, _CACHED_COLORED, _CACHED_METRICS, _GT_IMG, _P_CHK

    layer_idx = (
        int(layer_name.split()[-1]) if isinstance(layer_name, str) else int(layer_name)
    )

    puzzle_batch = _PUZZLE.reshape(_N, _L * _G1, _H, _W)
    puzzle_sample = _PUZZLE[sample_idx]
    sol_gt_raw = _SOLUTION[sample_idx]
    sol_gt_bin = ((sol_gt_raw + 1) / 2 > 0.5).float()

    puzzle_input = puzzle_batch[sample_idx : sample_idx + 1].to(_DEVICE)
    B, _, H, W = puzzle_input.shape
    out_c = _MODEL.conv_out.out_channels
    x = torch.randn(B, out_c, H, W, device=_DEVICE)
    dt = 1.0 / num_steps

    _P_CHK = puzzle_sample[layer_idx]
    _GT_IMG = render_colored(_P_CHK, sol_gt_bin[layer_idx], scale=8)

    heatmaps = []
    coloreds = []
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=_DEVICE)
        v = _MODEL(torch.cat([puzzle_input, x], dim=1), t)
        x = x + dt * v
        x_t = x[0].cpu()
        sol_t = (x_t[layer_idx] + 1) / 2

        label = f"Step {i + 1}/{num_steps}"

        h_img = render_heatmap(_P_CHK, x_t[layer_idx], scale=8)
        heatmaps.append(concat_images(h_img, _GT_IMG, label_left=label, label_right="Ground Truth"))

        c_img = render_colored(_P_CHK, sol_t, scale=8)
        coloreds.append(concat_images(c_img, _GT_IMG, label_left=label, label_right="Ground Truth"))

    sol_pred_bin = (sol_t > 0.5).float()
    sol_gt_layer = sol_gt_bin[layer_idx]
    correct = (sol_pred_bin == sol_gt_layer).sum().item()
    total = sol_gt_layer.numel()
    pixel_acc = correct / total
    intersection = (sol_pred_bin * sol_gt_layer).sum().item()
    union = ((sol_pred_bin + sol_gt_layer) > 0).sum().item()
    iou = intersection / union if union > 0 else 0.0

    _CACHED_HEATMAP = heatmaps
    _CACHED_COLORED = coloreds
    _CACHED_METRICS = {"acc": pixel_acc, "iou": iou}

    total_steps = len(heatmaps)
    return (
        coloreds[-1],
        f"{pixel_acc:.4f}",
        f"{iou:.4f}",
        gr.Slider(value=total_steps - 1, maximum=total_steps - 1, step=1, label="Step"),
        "colored",
    )


def show_frame(step_idx, mode):
    cache = _CACHED_COLORED if mode == "colored" else _CACHED_HEATMAP
    if cache:
        idx = min(max(int(step_idx), 0), len(cache) - 1)
        return cache[idx]
    return None


def on_mode_change(mode, step_val):
    return show_frame(int(step_val), mode)


def build_ui():
    with gr.Blocks(title="Maze Denoising Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Maze Denoising Explorer")
        gr.Markdown("Run denoising, then step through the animation to see the solution emerge.")

        with gr.Row():
            with gr.Column(scale=1):
                sample_slider = gr.Slider(0, _N - 1, step=1, value=0, label="Sample Index")
                layer_radio = gr.Radio([f"Layer {i}" for i in range(_L)], value="Layer 0", label="Layer")
                steps_slider = gr.Slider(2, 200, step=1, value=50, label="Total Denoise Steps")

                denoise_btn = gr.Button("Denoise", variant="primary", size="lg")

                with gr.Row():
                    acc_box = gr.Textbox(label="Pixel Accuracy", interactive=False)
                    iou_box = gr.Textbox(label="IoU", interactive=False)

                mode_radio = gr.Radio(
                    ["colored", "raw"],
                    value="colored",
                    label="Display Mode",
                )

            with gr.Column(scale=2):
                output_img = gr.Image(label="Result", type="pil")

        gr.Markdown("---")
        gr.Markdown("### Step Animation")
        with gr.Row():
            back_btn = gr.Button("◀")
            next_btn = gr.Button("▶")
            play_btn = gr.Button("Play ▶▶", variant="primary")
        step_slider = gr.Slider(0, 1, step=1, value=0, label="Step", interactive=True)

        gr.Markdown("---")
        gr.Markdown(
            "**Tips**: Click **Denoise** first. Use **Display Mode** to switch between raw heatmap and colored routes. "
            "Then use Play or the slider to step through."
        )

        def step_fwd(current):
            n = len(_CACHED_COLORED)
            if n == 0:
                return 0
            return min(current + 1, n - 1)

        def step_back(current):
            n = len(_CACHED_COLORED)
            if n == 0:
                return 0
            return max(current - 1, 0)

        def on_step_change(step_val, mode):
            return show_frame(int(step_val), mode)

        def on_play_click(progress=gr.Progress()):
            total = len(_CACHED_COLORED)
            if total == 0:
                return 0
            for i in range(total):
                progress((i + 1) / total)
                time.sleep(0.08)
                yield i
            yield total - 1

        denoise_btn.click(
            fn=run_denoise,
            inputs=[sample_slider, layer_radio, steps_slider],
            outputs=[output_img, acc_box, iou_box, step_slider, mode_radio],
        )

        step_slider.change(
            fn=on_step_change,
            inputs=[step_slider, mode_radio],
            outputs=[output_img],
            show_progress="hidden",
        )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio, step_slider],
            outputs=[output_img],
            show_progress="hidden",
        )

        next_btn.click(fn=step_fwd, inputs=[step_slider], outputs=[step_slider], show_progress="hidden")
        back_btn.click(fn=step_back, inputs=[step_slider], outputs=[step_slider], show_progress="hidden")
        play_btn.click(fn=on_play_click, outputs=[step_slider], show_progress="hidden")

    return demo


def main():
    parser = argparse.ArgumentParser(description="Interactive denoising visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="maze.safetensors")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    global _MODEL, _PUZZLE, _SOLUTION, _N, _L, _G1, _H, _W

    print("Loading model...")
    _MODEL, _ = load_model(args.checkpoint, _DEVICE)
    _MODEL.eval()

    print("Loading data...")
    data = load_file(args.data_path)
    _PUZZLE = data["puzzle"].float()
    _SOLUTION = data["solution"].float()
    _N, _L, _G1, _H, _W = _PUZZLE.shape
    print(f"Dataset: {_N} samples, {_L} layers, {_G1 - 1} routes, {_H}x{_W}")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()