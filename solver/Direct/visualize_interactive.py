import argparse
import os
import sys
import time

import gradio as gr
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import load_model
from utils.viz import render_colored, render_heatmap, concat_images

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHED_PRED = None
_CACHED_GT_IMG = None
_P_CHK = None


def predict_sample(sample_idx, layer_name):
    global _MODEL, _PUZZLE, _SOLUTION, _N, _L, _G1, _H, _W
    global _CACHED_PRED, _CACHED_GT_IMG, _P_CHK

    layer_idx = (
        int(layer_name.split()[-1]) if isinstance(layer_name, str) else int(layer_name)
    )

    puzzle_batch = _PUZZLE.reshape(_N, _L * _G1, _H, _W)
    puzzle_input = puzzle_batch[sample_idx : sample_idx + 1].to(_DEVICE)
    sol_gt_raw = _SOLUTION[sample_idx]
    sol_gt_bin = ((sol_gt_raw + 1) / 2 > 0.5).float()

    with torch.no_grad():
        pred = _MODEL(puzzle_input)

    pred = pred[0].cpu()
    sol_pred_bin = ((pred + 1) / 2 > 0.5).float()
    _CACHED_PRED = sol_pred_bin

    puzzle_sample = _PUZZLE[sample_idx]
    _P_CHK = puzzle_sample[layer_idx]
    _CACHED_GT_IMG = render_colored(_P_CHK, sol_gt_bin[layer_idx], scale=8)

    pred_img = render_colored(_P_CHK, sol_pred_bin[layer_idx], scale=8)
    heatmap_img = render_heatmap(_P_CHK, pred[layer_idx], scale=8)

    correct = (sol_pred_bin[layer_idx] == sol_gt_bin[layer_idx]).sum().item()
    total = sol_gt_bin[layer_idx].numel()
    pixel_acc = correct / total
    intersection = (sol_pred_bin[layer_idx] * sol_gt_bin[layer_idx]).sum().item()
    union = ((sol_pred_bin[layer_idx] + sol_gt_bin[layer_idx]) > 0).sum().item()
    iou = intersection / union if union > 0 else 0.0

    combined = concat_images(pred_img, _CACHED_GT_IMG, label_left="Prediction", label_right="Ground Truth")

    return combined, f"{pixel_acc:.4f}", f"{iou:.4f}"


def show_heatmap():
    if _CACHED_PRED is not None and _P_CHK is not None:
        pred_raw = _CACHED_PRED
        heatmap_img = render_heatmap(_P_CHK, pred_raw[0] if pred_raw.ndim == 3 else pred_raw, scale=8)
        return concat_images(heatmap_img, _CACHED_GT_IMG, label_left="Heatmap", label_right="Ground Truth")
    return None


def build_ui():
    with gr.Blocks(title="Direct Maze Solver", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Direct Maze Solver")
        gr.Markdown("Predict maze solutions directly from puzzle input.")

        with gr.Row():
            with gr.Column(scale=1):
                sample_slider = gr.Slider(0, _N - 1, step=1, value=0, label="Sample Index")
                layer_radio = gr.Radio([f"Layer {i}" for i in range(_L)], value="Layer 0", label="Layer")
                predict_btn = gr.Button("Predict", variant="primary", size="lg")
                heatmap_btn = gr.Button("Show Heatmap")

                with gr.Row():
                    acc_box = gr.Textbox(label="Pixel Accuracy", interactive=False)
                    iou_box = gr.Textbox(label="IoU", interactive=False)

            with gr.Column(scale=2):
                output_img = gr.Image(label="Result", type="pil")

        predict_btn.click(
            fn=predict_sample,
            inputs=[sample_slider, layer_radio],
            outputs=[output_img, acc_box, iou_box],
        )

        heatmap_btn.click(
            fn=show_heatmap,
            outputs=[output_img],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Interactive direct solver visualization")
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
    _SOLUTION = 2.0 * _SOLUTION - 1.0
    _N, _L, _G1, _H, _W = _PUZZLE.shape
    print(f"Dataset: {_N} samples, {_L} layers, {_G1 - 1} routes, {_H}x{_W}")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()