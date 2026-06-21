import argparse
import csv
import json
import platform
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from absa_pipeline import (
    ABSAClassifier,
    LABELS,
    PIPELINE_VERSION,
    build_full_train_loaders,
    build_loaders,
    build_optimizer,
    describe_samples,
    evaluate,
    fit_temperature,
    set_seed,
    train_fixed_epochs,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an ABSA classifier")
    parser.add_argument("--dataset", choices=["lap", "rest", "twi"], default="rest")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "linear"], default="none")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--input-format",
        choices=["term", "term_desc"],
        default="term",
        help="Use raw aspect term only, or append aspect description as a separate ablation.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--loss-type",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt", "balanced"],
        default="none",
    )
    parser.add_argument(
        "--neutral-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiply only the neutral class weight, then renormalize mean class weight to 1.",
    )
    parser.add_argument(
        "--pooling-strategy",
        choices=["cls", "aspect_attention", "hybrid"],
        default="cls",
    )
    parser.add_argument(
        "--encoder-layer-pooling",
        choices=["last", "mean_last4"],
        default="last",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.002)
    parser.add_argument(
        "--temperature-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--refit-full-train", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--evaluation-split",
        choices=["test", "validation"],
        default="test",
        help="Use validation during tuning; reserve test for the selected configuration.",
    )
    return parser.parse_args()


def save_history(history, output_dir, run_name):
    csv_path = output_dir / f"{run_name}_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    figure_path = output_dir / f"{run_name}_learning_curves.png"
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["valid_accuracy"] for row in history], marker="o", label="accuracy")
    axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], marker="o", label="macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation classification")
    axes[1].legend()

    axes[2].plot(epochs, [row["valid_auc_macro"] for row in history], marker="o", color="#c56")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Validation AUC")

    axes[3].plot(epochs, [row["valid_avg_confidence"] for row in history], marker="o", color="#2a7")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Validation confidence")

    axes[4].plot(epochs, [row["valid_weighted_f1"] for row in history], marker="o", color="#805ad5")
    axes[4].set_ylim(0, 1)
    axes[4].set_title("Validation weighted F1")

    axes[5].plot(epochs, [row["learning_rate"] for row in history], color="#dd6b20")
    axes[5].set_title("Encoder learning rate")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return [csv_path, figure_path]


def save_confusion_matrix(matrix, labels, output_dir, run_name, evaluation_name):
    path = output_dir / f"{run_name}_confusion_matrix.png"
    counts = np.asarray(matrix)
    normalized = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, title, value_format in (
        (axes[0], counts, f"{evaluation_name} confusion matrix (counts)", "d"),
        (
            axes[1],
            normalized,
            f"{evaluation_name} confusion matrix (row normalized)",
            ".2f",
        ),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks(range(len(labels)), labels=labels, rotation=30)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = int(values[row, column]) if value_format == "d" else values[row, column]
                axis.text(column, row, format(value, value_format), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_predictions(prediction_rows, samples, output_dir, run_name):
    path = output_dir / f"{run_name}_predictions.csv"
    fieldnames = [
        "sentence",
        "aspect",
        "gold_label",
        "predicted_label",
        "confidence",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "is_seen",
        "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample, prediction in zip(samples, prediction_rows):
            writer.writerow(
                {
                    "sentence": sample["sentence"],
                    "aspect": sample["aspect"],
                    **prediction,
                }
            )
    return path


def save_metrics_table(metrics, output_dir, run_name):
    path = output_dir / f"{run_name}_metrics.csv"
    rows = [
        {"scope": "overall", "metric": "accuracy", "value": metrics["accuracy"]},
        {
            "scope": "overall",
            "metric": "balanced_accuracy",
            "value": metrics["balanced_accuracy"],
        },
        {"scope": "overall", "metric": "loss", "value": metrics["loss"]},
        {"scope": "overall", "metric": "log_loss", "value": metrics["log_loss"]},
        {
            "scope": "overall",
            "metric": "roc_auc_macro",
            "value": metrics["roc_auc"]["macro"],
        },
        {
            "scope": "overall",
            "metric": "average_confidence",
            "value": metrics["confidence"]["average"],
        },
        {
            "scope": "overall",
            "metric": "expected_calibration_error",
            "value": metrics["confidence"]["expected_calibration_error"],
        },
        {
            "scope": "overall",
            "metric": "brier_score",
            "value": metrics["confidence"]["brier_score"],
        },
    ]
    for average, values in metrics["aggregate_metrics"].items():
        for metric_name, value in values.items():
            rows.append(
                {
                    "scope": average,
                    "metric": metric_name,
                    "value": value,
                }
            )
    for label, values in metrics["per_class"].items():
        for metric_name, value in values.items():
            rows.append(
                {
                    "scope": label,
                    "metric": metric_name,
                    "value": value,
                }
            )
        rows.append(
            {
                "scope": label,
                "metric": "roc_auc",
                "value": metrics["roc_auc"]["per_class"].get(label),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_roc_curve(metrics, output_dir, run_name):
    path = output_dir / f"{run_name}_roc_curve.png"
    if not metrics.get("roc_curve"):
        return None
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for label, data in metrics["roc_curve"].items():
        auc_text = "n/a" if data["auc"] is None else f"{data['auc']:.3f}"
        axis.plot(data["fpr"], data["tpr"], label=f"{label} (AUC={auc_text})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("One-vs-rest ROC curves")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_confidence_histogram(prediction_rows, output_dir, run_name):
    path = output_dir / f"{run_name}_confidence_histogram.png"
    correct = [row["confidence"] for row in prediction_rows if row["correct"]]
    wrong = [row["confidence"] for row in prediction_rows if not row["correct"]]
    figure, axis = plt.subplots(figsize=(6.5, 5))
    axis.hist(correct, bins=20, alpha=0.7, label="correct")
    axis.hist(wrong, bins=20, alpha=0.7, label="wrong")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Count")
    axis.set_title("Prediction confidence distribution")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_class_metrics(metrics, output_dir, run_name, evaluation_name):
    path = output_dir / f"{run_name}_class_metrics.png"
    x = np.arange(len(LABELS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 5))
    for offset, metric, color in (
        (-width, "precision", "#3182ce"),
        (0, "recall", "#38a169"),
        (width, "f1", "#dd6b20"),
    ):
        values = [metrics["per_class"][label][metric] for label in LABELS]
        axis.bar(x + offset, values, width, label=metric.title(), color=color)
    axis.set_xticks(x, LABELS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"Per-class {evaluation_name.lower()} metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_calibration_curve(metrics, output_dir, run_name):
    path = output_dir / f"{run_name}_calibration_curve.png"
    bins = metrics["confidence"]["calibration_bins"]
    if not bins:
        return None
    confidence = [row["confidence"] for row in bins]
    accuracy = [row["accuracy"] for row in bins]
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(confidence, accuracy, marker="o", color="#2b6cb0", label="Model")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed accuracy")
    axis.set_title(
        "Reliability diagram "
        f"(ECE={metrics['confidence']['expected_calibration_error']:.3f})"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def class_weights_for(samples, weighting, neutral_weight_multiplier=1.0):
    if neutral_weight_multiplier <= 0:
        raise ValueError("neutral_weight_multiplier must be positive")
    if weighting == "none":
        weights = np.ones(len(LABELS), dtype=np.float64)
    else:
        counts = np.bincount(
            [sample["label"] for sample in samples], minlength=len(LABELS)
        ).astype(np.float64)
        weights = counts.sum() / (len(LABELS) * counts)
        if weighting == "sqrt":
            weights = np.sqrt(weights)
    # Neutral is label index 1. Normalize after scaling so the average loss
    # magnitude stays comparable across ablations.
    weights[LABELS.index("neutral")] *= neutral_weight_multiplier
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def safe_run_name(args):
    if args.run_name:
        raw_name = args.run_name
    else:
        model_name = args.model.rsplit("/", 1)[-1]
        raw_name = f"{args.dataset}_{model_name}_seed{args.seed}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    if not name:
        raise ValueError("run name must contain at least one letter or number")
    return name


def main():
    args = parse_args()
    print(f"pipeline_version={PIPELINE_VERSION}")
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data_dir = Path("dataset") / args.dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = safe_run_name(args)
    checkpoint = output_dir / f"{run_name}.pt"
    if args.evaluation_split == "validation" and args.refit_full_train:
        raise ValueError("--refit-full-train cannot be used with validation evaluation")

    train_loader, valid_loader, test_loader, splits = build_loaders(
        data_dir=data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        valid_ratio=args.valid_ratio,
        split_seed=args.split_seed,
        input_format=args.input_format,
    )
    print(json.dumps({"device": str(device), "data": describe_samples(splits)}, indent=2))

    class_weights = class_weights_for(
        splits["train"],
        args.class_weighting,
        neutral_weight_multiplier=args.neutral_weight_multiplier,
    )
    print(json.dumps({"class_weights": dict(zip(LABELS, [round(float(x), 6) for x in class_weights]))}, indent=2))

    def create_model(current_weights):
        model = ABSAClassifier(
            model_name=args.model,
            dropout=args.dropout,
            class_weights=current_weights,
            label_smoothing=args.label_smoothing,
            pooling_strategy=args.pooling_strategy,
            encoder_layer_pooling=args.encoder_layer_pooling,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
        ).to(device)
        if args.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return model

    def optimizer_builder(model):
        return build_optimizer(
            model,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
        )

    model = create_model(class_weights)
    model, training = train_model(
        model,
        train_loader,
        valid_loader,
        device,
        args.epochs,
        checkpoint,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        use_amp=args.amp,
        optimizer_builder=optimizer_builder,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    temperature = 1.0
    if args.temperature_scaling:
        temperature = fit_temperature(
            model,
            valid_loader,
            device,
            use_amp=args.amp,
        )
        print(f"calibration_temperature={temperature:.6f}")

    if args.refit_full_train:
        selected_epochs = training["best_epoch"]
        set_seed(args.seed)
        full_train_loader, test_loader, full_splits = build_full_train_loaders(
            data_dir=data_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            input_format=args.input_format,
        )
        full_weights = class_weights_for(
            full_splits["train"],
            args.class_weighting,
            neutral_weight_multiplier=args.neutral_weight_multiplier,
        )
        model = create_model(full_weights)
        model, refit = train_fixed_epochs(
            model,
            full_train_loader,
            device,
            selected_epochs,
            checkpoint,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            use_amp=args.amp,
            optimizer_builder=optimizer_builder,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            scheduler_total_epochs=args.epochs,
        )
        training["refit"] = refit

    evaluation_loader = valid_loader if args.evaluation_split == "validation" else test_loader
    evaluation_name = args.evaluation_split.title()
    raw_metrics = evaluate(
        model,
        evaluation_loader,
        device,
        use_amp=args.amp,
        include_predictions=True,
    )
    metrics = (
        evaluate(
            model,
            evaluation_loader,
            device,
            use_amp=args.amp,
            include_predictions=True,
            temperature=temperature,
        )
        if args.temperature_scaling
        else raw_metrics
    )
    prediction_rows = metrics.pop("prediction_rows")
    raw_metrics.pop("prediction_rows", None)
    metrics.update(
        {
            "labels": LABELS,
            "data": describe_samples(splits),
            "training": training,
            "config": vars(args),
            "pipeline_version": PIPELINE_VERSION,
            "evaluation_split": args.evaluation_split,
            "calibration": {
                "temperature": round(temperature, 6),
                "uncalibrated_log_loss": raw_metrics["log_loss"],
                "uncalibrated_confidence": raw_metrics["confidence"],
            },
            "runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
        }
    )
    result_path = output_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifacts = [checkpoint, result_path]
    artifacts.extend(save_history(training["history"], output_dir, run_name))
    artifacts.append(save_metrics_table(metrics, output_dir, run_name))
    artifacts.append(
        save_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["labels"],
            output_dir,
            run_name,
            evaluation_name,
        )
    )
    artifacts.append(save_class_metrics(metrics, output_dir, run_name, evaluation_name))
    roc_path = save_roc_curve(metrics, output_dir, run_name)
    if roc_path is not None:
        artifacts.append(roc_path)
    artifacts.append(save_confidence_histogram(prediction_rows, output_dir, run_name))
    calibration_path = save_calibration_curve(metrics, output_dir, run_name)
    if calibration_path is not None:
        artifacts.append(calibration_path)
    artifacts.append(
        save_predictions(
            prediction_rows,
            evaluation_loader.dataset.samples,
            output_dir,
            run_name,
        )
    )

    bundle = output_dir / f"{run_name}_artifacts.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            if artifact is not None:
                archive.write(artifact, arcname=artifact.name)

    print(
        json.dumps(
            {
                args.evaluation_split: {
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "roc_auc_macro": metrics["roc_auc"]["macro"],
                    "average_confidence": metrics["confidence"]["average"],
                    "expected_calibration_error": metrics["confidence"]["expected_calibration_error"],
                    "temperature": metrics["calibration"]["temperature"],
                    "uncalibrated_confidence": metrics["calibration"][
                        "uncalibrated_confidence"
                    ]["average"],
                    "seen": metrics["seen"],
                    "unseen": metrics["unseen"],
                },
                "zip": str(bundle),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
