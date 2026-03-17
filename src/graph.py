import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)\.pth$")


@dataclass
class ScalarSeries:
    name: str
    steps: list[int]
    values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all scalar TensorBoard data for an experiment."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/hand11_bs16_lr5e5"),
        help="Experiment directory containing logs/ and checkpoints/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <experiment-dir>/all_data_visualization.png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <experiment-dir>/all_data_by_epoch.csv.",
    )
    return parser.parse_args()


def iter_event_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.rglob("events.out.tfevents.*"))


def normalize_run_label(log_dir: Path, event_file: Path) -> str:
    parent = event_file.parent
    if parent == log_dir:
        return "main"
    return parent.relative_to(log_dir).as_posix()


def scalar_from_value(value) -> float | None:
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return float(value.simple_value)
    if kind != "tensor":
        return None

    array = tensor_util.make_ndarray(value.tensor)
    if array.size != 1:
        return None
    return float(array.reshape(-1)[0])


def load_scalar_series(log_dir: Path) -> list[ScalarSeries]:
    series_points: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for event_file in iter_event_files(log_dir):
        run_label = normalize_run_label(log_dir, event_file)
        for event in EventFileLoader(str(event_file)).Load():
            if not event.summary.value:
                continue
            for value in event.summary.value:
                scalar = scalar_from_value(value)
                if scalar is None:
                    continue
                series_name = (
                    value.tag if run_label == "main" else f"{run_label} | {value.tag}"
                )
                series_points[series_name].append((int(event.step), scalar))

    series: list[ScalarSeries] = []
    for name in sorted(series_points):
        deduped = sorted(set(series_points[name]), key=lambda item: item[0])
        steps = [step for step, _ in deduped]
        values = [value for _, value in deduped]
        series.append(ScalarSeries(name=name, steps=steps, values=values))
    return series


def load_checkpoint_epochs(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.exists():
        return []

    epochs: list[int] = []
    for checkpoint in sorted(checkpoint_dir.glob("checkpoint_epoch*.pth")):
        match = CHECKPOINT_RE.search(checkpoint.name)
        if match:
            epochs.append(int(match.group(1)))
    return epochs


def sanitize_column_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return sanitized or "series"


def infer_steps_per_epoch(epoch_series: ScalarSeries) -> int:
    per_epoch_counts: dict[int, int] = defaultdict(int)
    for value in epoch_series.values:
        per_epoch_counts[int(round(value))] += 1
    if not per_epoch_counts:
        raise RuntimeError("Could not infer steps per epoch from the Epoch scalar.")
    return max(per_epoch_counts.values())


def map_step_to_epoch(step: int, steps_per_epoch: int) -> int:
    return max(1, int(math.ceil(step / steps_per_epoch)))


def map_step_to_epoch_position(step: int, steps_per_epoch: int) -> float:
    return step / steps_per_epoch


def build_epoch_rows(
    scalar_series: list[ScalarSeries],
    checkpoint_epochs: list[int],
) -> tuple[list[dict[str, float | int | str]], list[str]]:
    series_by_name = {series.name: series for series in scalar_series}
    epoch_series = series_by_name.get("Epoch")
    if epoch_series is None:
        raise RuntimeError("Epoch scalar is required to build the epoch CSV.")

    steps_per_epoch = infer_steps_per_epoch(epoch_series)
    epoch_end_steps: dict[int, int] = {}
    for step in epoch_series.steps:
        epoch = map_step_to_epoch(step, steps_per_epoch)
        epoch_end_steps[epoch] = max(epoch_end_steps.get(epoch, 0), step)

    aggregated: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for series in scalar_series:
        if series.name == "Epoch":
            continue
        for step, value in zip(series.steps, series.values):
            epoch = map_step_to_epoch(step, steps_per_epoch)
            aggregated[epoch][series.name].append(value)
            epoch_end_steps[epoch] = max(epoch_end_steps.get(epoch, 0), step)

    all_epochs = sorted(epoch_end_steps)
    checkpoint_set = set(checkpoint_epochs)
    series_names = [series.name for series in scalar_series if series.name != "Epoch"]
    repeated_series = {
        name
        for name in series_names
        if any(len(aggregated[epoch].get(name, [])) > 1 for epoch in all_epochs)
    }

    fieldnames = ["epoch", "epoch_end_step", "checkpoint_saved"]
    for name in series_names:
        column_base = sanitize_column_name(name)
        if name in repeated_series:
            fieldnames.extend(
                [f"{column_base}_last", f"{column_base}_mean", f"{column_base}_count"]
            )
        else:
            fieldnames.append(column_base)

    rows: list[dict[str, float | int | str]] = []
    for epoch in all_epochs:
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "epoch_end_step": epoch_end_steps[epoch],
            "checkpoint_saved": int(epoch in checkpoint_set),
        }
        for name in series_names:
            values = aggregated[epoch].get(name, [])
            if not values:
                continue
            column_base = sanitize_column_name(name)
            if name in repeated_series:
                row[f"{column_base}_last"] = values[-1]
                row[f"{column_base}_mean"] = sum(values) / len(values)
                row[f"{column_base}_count"] = len(values)
            else:
                row[column_base] = values[-1]
        rows.append(row)

    return rows, fieldnames


def write_epoch_csv(
    scalar_series: list[ScalarSeries],
    checkpoint_epochs: list[int],
    csv_output_path: Path,
) -> None:
    rows, fieldnames = build_epoch_rows(scalar_series, checkpoint_epochs)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_series(ax, series: ScalarSeries, steps_per_epoch: int) -> None:
    epoch_positions = [
        map_step_to_epoch_position(step, steps_per_epoch) for step in series.steps
    ]

    ax.plot(epoch_positions, series.values, color="#0f766e", linewidth=1.8)
    if len(epoch_positions) <= 20:
        ax.scatter(epoch_positions, series.values, color="#0f766e", s=18)

    ax.set_title(series.name, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)

    if series.values:
        ax.annotate(
            f"{series.values[-1]:.4f}",
            (epoch_positions[-1], series.values[-1]),
            textcoords="offset points",
            xytext=(-4, 8),
            ha="right",
            fontsize=8,
        )


def plot_checkpoints(ax, epochs: list[int]) -> None:
    ax.set_title("Checkpoints")
    if not epochs:
        ax.text(0.5, 0.5, "No checkpoints found", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(epochs, [1] * len(epochs), color="#b91c1c", s=48)
    for epoch in epochs:
        ax.annotate(
            f"e{epoch}",
            (epoch, 1),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Epoch")
    ax.set_yticks([])
    ax.set_ylim(0.9, 1.12)
    ax.grid(True, axis="x", alpha=0.3)


def render_figure(
    experiment_dir: Path,
    scalar_series: list[ScalarSeries],
    checkpoint_epochs: list[int],
    steps_per_epoch: int,
    output_path: Path,
) -> None:
    panel_count = len(scalar_series) + 1
    cols = min(3, max(1, math.ceil(math.sqrt(panel_count))))
    rows = math.ceil(panel_count / cols)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.0))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, series in zip(axes_list, scalar_series):
        plot_series(ax, series, steps_per_epoch)

    plot_checkpoints(axes_list[len(scalar_series)], checkpoint_epochs)

    for ax in axes_list[panel_count:]:
        ax.set_axis_off()

    fig.suptitle(f"Experiment Summary: {experiment_dir.as_posix()}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    log_dir = experiment_dir / "logs"
    checkpoint_dir = experiment_dir / "checkpoints"

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else experiment_dir / "all_data_visualization.png"
    )
    csv_output_path = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else experiment_dir / "all_data_by_epoch.csv"
    )

    scalar_series = load_scalar_series(log_dir)
    if not scalar_series:
        raise RuntimeError(f"No scalar data found in {log_dir}")

    checkpoint_epochs = load_checkpoint_epochs(checkpoint_dir)
    epoch_series = next((series for series in scalar_series if series.name == "Epoch"), None)
    if epoch_series is None:
        raise RuntimeError("Epoch scalar is required to render the epoch x-axis.")
    steps_per_epoch = infer_steps_per_epoch(epoch_series)

    render_figure(
        experiment_dir,
        scalar_series,
        checkpoint_epochs,
        steps_per_epoch,
        output_path,
    )
    write_epoch_csv(scalar_series, checkpoint_epochs, csv_output_path)

    print(f"Saved visualization to {output_path}")
    print(f"Saved epoch CSV to {csv_output_path}")
    print(f"Plotted {len(scalar_series)} scalar series")


if __name__ == "__main__":
    main()
