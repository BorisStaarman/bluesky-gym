from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


IMAGE_NAME = "training_metrics_run_1.png"
EVAL_CSV = Path("metrics") / "run_1" / "evaluation_progress.csv"
NUM_PANELS = 6
BACKGROUND_THRESHOLD = 245


def find_panel_boundaries(image_array: np.ndarray, num_panels: int) -> list[int]:
    mask = (image_array < BACKGROUND_THRESHOLD).any(axis=2)
    row_counts = mask.sum(axis=1).astype(float)

    window = max(31, image_array.shape[0] // 120)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    smooth = np.convolve(row_counts, kernel, mode="same")

    search_radius = max(150, image_array.shape[0] // 28)
    boundaries = [0]
    for index in range(1, num_panels):
        expected = int(image_array.shape[0] * index / num_panels)
        lower = max(0, expected - search_radius)
        upper = min(image_array.shape[0], expected + search_radius)
        best = lower + int(np.argmin(smooth[lower:upper]))
        boundaries.append(best)
    boundaries.append(image_array.shape[0])
    return boundaries


def trim_segment(image_array: np.ndarray, top: int, bottom: int) -> tuple[int, int, int, int]:
    segment = image_array[top:bottom]
    mask = (segment < BACKGROUND_THRESHOLD).any(axis=2)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    row_threshold = max(10, image_array.shape[1] // 200)
    col_threshold = max(10, image_array.shape[0] // 250)

    row_indices = np.where(row_counts > row_threshold)[0]
    col_indices = np.where(col_counts > col_threshold)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return 0, top, image_array.shape[1], bottom

    pad_y = 28
    pad_x = 28
    left = max(0, int(col_indices[0]) - pad_x)
    right = min(image_array.shape[1], int(col_indices[-1]) + pad_x + 1)
    cropped_top = max(0, top + int(row_indices[0]) - pad_y)
    cropped_bottom = min(image_array.shape[0], top + int(row_indices[-1]) + pad_y + 1)
    return left, cropped_top, right, cropped_bottom


def extract_panel(image_path: Path, panel_number: int, output_name: str) -> Path:
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    boundaries = find_panel_boundaries(image_array, NUM_PANELS)

    panel_index = panel_number - 1
    top = boundaries[panel_index]
    bottom = boundaries[panel_index + 1]
    left, cropped_top, right, cropped_bottom = trim_segment(image_array, top, bottom)

    cropped = image.crop((left, cropped_top, right, cropped_bottom))
    output_path = image_path.with_name(output_name)
    cropped.save(output_path)
    return output_path


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def remove_red_series_and_update_title(image_path: Path, output_name: str) -> Path:
    cropped_path = extract_panel(image_path, 2, output_name)
    image = Image.open(cropped_path).convert("RGB")
    image_array = np.asarray(image).copy()

    red_mask = (
        (image_array[:, :, 0] > 150)
        & (image_array[:, :, 1] < 150)
        & (image_array[:, :, 2] < 150)
    )

    expanded_mask = red_mask.copy()
    for shift_y in (-2, -1, 0, 1, 2):
        for shift_x in (-2, -1, 0, 1, 2):
            shifted = np.zeros_like(red_mask)
            src_y_start = max(0, -shift_y)
            src_y_end = red_mask.shape[0] - max(0, shift_y)
            src_x_start = max(0, -shift_x)
            src_x_end = red_mask.shape[1] - max(0, shift_x)
            dst_y_start = max(0, shift_y)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_start = max(0, shift_x)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = red_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            expanded_mask |= shifted

    image_array[expanded_mask] = 255
    updated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(updated)

    width, height = updated.size

    title_box = (int(0.18 * width), 0, int(0.83 * width), int(0.11 * height))
    draw.rectangle(title_box, fill="white")
    title_font = load_font(max(24, width // 55))
    title_text = "Training reward"
    text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_x = (width - (text_bbox[2] - text_bbox[0])) // 2
    title_y = max(10, int(0.025 * height))
    draw.text((title_x, title_y), title_text, fill="black", font=title_font)

    legend_box = (int(0.11 * width), int(0.12 * height), int(0.39 * width), int(0.23 * height))
    draw.rectangle(legend_box, fill="white")
    legend_font = load_font(max(20, width // 80))
    line_y = int(0.165 * height)
    line_x1 = int(0.14 * width)
    line_x2 = int(0.19 * width)
    draw.line((line_x1, line_y, line_x2, line_y), fill=(70, 130, 180), width=max(3, width // 700))
    draw.text((int(0.205 * width), int(0.145 * height)), "Mean Reward", fill="black", font=legend_font)

    updated.save(cropped_path)
    return cropped_path


def generate_evaluation_plot(script_dir: Path, output_name: str) -> Path:
    csv_path = script_dir / EVAL_CSV
    df = pd.read_csv(csv_path).sort_values("iteration").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_intrusions = "crimson"
    color_waypoints = "darkgreen"

    intrusions_line, = ax1.plot(
        df["iteration"],
        df["avg_intrusions"],
        marker="o",
        linewidth=2,
        color=color_intrusions,
        label="Mean Intrusions",
    )
    ax1.set_xlabel("Training Iteration", fontsize=12)
    ax1.set_ylabel("Mean Intrusions", color=color_intrusions, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_intrusions)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    waypoint_line, = ax2.plot(
        df["iteration"],
        df["waypoint_rate"] * 100.0,
        marker="s",
        linewidth=2,
        color=color_waypoints,
        label="Waypoint Success Rate (%)",
    )
    ax2.set_ylabel("Waypoint Success Rate (%)", color=color_waypoints, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_waypoints)

    ax1.set_title("Evaluation during training", fontsize=14, fontweight="bold")
    ax1.legend(
        [intrusions_line, waypoint_line],
        [intrusions_line.get_label(), waypoint_line.get_label()],
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
    )

    fig.tight_layout()
    output_path = script_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    image_path = script_dir / IMAGE_NAME

    outputs = [
        (2, "training_metrics_run_1_reward_only.png"),
        (6, "training_metrics_run_1_evaluation_only.png"),
    ]

    for panel_number, output_name in outputs:
        output_path = extract_panel(image_path, panel_number, output_name)
        print(f"Saved panel {panel_number} to: {output_path}")


if __name__ == "__main__":
    main()