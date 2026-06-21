import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INITIAL_SEARCH_SPACE = [
    {
        "name": "cls_reference",
        "pooling_strategy": "cls",
        "encoder_learning_rate": 1e-5,
        "head_learning_rate": 1e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.0,
        "scheduler": "none",
        "warmup_ratio": 0.0,
        "max_length": 128,
    },
    {
        "name": "cls_linear_decay",
        "pooling_strategy": "cls",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.0,
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_class_balanced",
        "pooling_strategy": "cls",
        "encoder_learning_rate": 1e-5,
        "head_learning_rate": 1e-5,
        "dropout": 0.2,
        "class_weighting": "sqrt",
        "label_smoothing": 0.02,
        "scheduler": "none",
        "warmup_ratio": 0.0,
        "max_length": 128,
    },
    {
        "name": "aspect_attention",
        "pooling_strategy": "aspect_attention",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 3e-5,
        "dropout": 0.2,
        "class_weighting": "sqrt",
        "label_smoothing": 0.02,
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 192,
    },
    {
        "name": "hybrid_pooling",
        "pooling_strategy": "hybrid",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 3e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.02,
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 192,
    },
]


REFINED_SEARCH_SPACE = [
    {
        "name": "cls_linear_anchor",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.0,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_linear_smooth",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_linear_sqrt",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "sqrt",
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_linear_dropout",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_linear_focal",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.0,
        "loss_type": "focal",
        "focal_gamma": 1.5,
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_mean_last4",
        "pooling_strategy": "cls",
        "encoder_layer_pooling": "mean_last4",
        "encoder_learning_rate": 6e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.2,
        "class_weighting": "none",
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
]


# Clean ablation spaces for the neutral/unseen investigation.
# Keep input_format=term for the core space so pooling and neutral weighting are
# not confounded with generated aspect descriptions. Description trials live in
# a separate space and should be interpreted as exploratory.
ABLATION_CORE_SEARCH_SPACE = [
    {
        "name": "baseline_cls",
        "pooling_strategy": "cls",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.0,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "aspect_attention_term",
        "pooling_strategy": "aspect_attention",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.0,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "hybrid_term",
        "pooling_strategy": "hybrid",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.0,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_neutral125",
        "pooling_strategy": "cls",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.25,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "cls_neutral150",
        "pooling_strategy": "cls",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.5,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
    {
        "name": "aspect_attention_neutral125",
        "pooling_strategy": "aspect_attention",
        "input_format": "term",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.25,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 128,
    },
]


ABLATION_DESC_SEARCH_SPACE = [
    {
        "name": "cls_desc",
        "pooling_strategy": "cls",
        "input_format": "term_desc",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.0,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 192,
    },
    {
        "name": "aspect_attention_desc",
        "pooling_strategy": "aspect_attention",
        "input_format": "term_desc",
        "encoder_layer_pooling": "last",
        "encoder_learning_rate": 8e-6,
        "head_learning_rate": 2e-5,
        "dropout": 0.3,
        "class_weighting": "none",
        "neutral_weight_multiplier": 1.0,
        "label_smoothing": 0.02,
        "loss_type": "cross_entropy",
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "max_length": 192,
    },
]


ABLATION_ALL_SEARCH_SPACE = ABLATION_CORE_SEARCH_SPACE + ABLATION_DESC_SEARCH_SPACE


SEARCH_SPACES = {
    "initial": INITIAL_SEARCH_SPACE,
    "refined": REFINED_SEARCH_SPACE,
    "ablation_core": ABLATION_CORE_SEARCH_SPACE,
    "ablation_desc": ABLATION_DESC_SEARCH_SPACE,
    "ablation_all": ABLATION_ALL_SEARCH_SPACE,
}


def parse_seed_list(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def parse_seed_pairs(value):
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            split_seed, model_seed = item.split(":", 1)
            pairs.append((int(split_seed), int(model_seed)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "Seed pairs must use split_seed:model_seed format"
            ) from error
    if not pairs:
        raise argparse.ArgumentTypeError("At least one validation seed pair is required")
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select ABSA hyperparameters using validation Macro-F1"
    )
    parser.add_argument("--dataset", choices=["lap", "rest", "twi"], default="rest")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--tuning-seeds", type=parse_seed_list, default=[2024, 42])
    parser.add_argument(
        "--validation-seed-pairs",
        type=parse_seed_pairs,
        default=None,
        help="Comma-separated split_seed:model_seed pairs.",
    )
    parser.add_argument("--final-seeds", type=parse_seed_list, default=[2024, 42, 3407])
    parser.add_argument("--output-dir", default="tuning_outputs")
    parser.add_argument("--final-output-dir", default="outputs")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--keep-trial-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--search-space",
        choices=sorted(SEARCH_SPACES),
        default="refined",
    )
    parser.add_argument(
        "--trials",
        default="all",
        help="Comma-separated trial names or 'all'.",
    )
    return parser.parse_args()


def selected_trials(value, search_space):
    available_trials = SEARCH_SPACES[search_space]
    if value == "all":
        return available_trials
    names = {item.strip() for item in value.split(",") if item.strip()}
    trials = [trial for trial in available_trials if trial["name"] in names]
    missing = names - {trial["name"] for trial in trials}
    if missing:
        raise ValueError(f"Unknown trial names: {', '.join(sorted(missing))}")
    if not trials:
        raise ValueError("No tuning trials selected")
    return trials


def option_name(key):
    return "--" + key.replace("_", "-")


def training_command(
    args,
    config,
    seed,
    output_dir,
    run_name,
    evaluation_split,
    epochs=None,
    split_seed=None,
):
    command = [
        sys.executable,
        "train.py",
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs if epochs is None else epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--weight-decay",
        str(args.weight_decay),
        "--valid-ratio",
        str(args.valid_ratio),
        "--split-seed",
        str(args.split_seed if split_seed is None else split_seed),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--evaluation-split",
        evaluation_split,
    ]
    for key, value in config.items():
        if key != "name":
            command.extend([option_name(key), str(value)])
    if args.amp:
        command.append("--amp")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def remove_large_trial_files(output_dir, run_name):
    for suffix in (".pt", "_artifacts.zip"):
        path = output_dir / f"{run_name}{suffix}"
        if path.exists():
            path.unlink()


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def save_tuning_plot(summary_rows, output_dir):
    ordered = list(reversed(summary_rows))
    labels = [row["trial"] for row in ordered]
    f1_values = [row["mean_macro_f1"] for row in ordered]
    f1_errors = [row["std_macro_f1"] for row in ordered]
    accuracy_values = [row["mean_accuracy"] for row in ordered]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(positions, f1_values, xerr=f1_errors, color="#3182ce", alpha=0.85)
    axes[0].set_yticks(positions, labels=labels)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Mean validation Macro-F1")
    axes[0].set_title("Hyperparameter ranking")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(positions, accuracy_values, color="#38a169", alpha=0.85)
    axes[1].set_yticks(positions, labels=labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Mean validation Accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "tuning_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_final_summary(final_rows, output_dir):
    csv_path = output_dir / "final_runs_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)

    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "roc_auc_macro",
        "log_loss",
        "expected_calibration_error",
    ]
    aggregate = {
        metric: {
            "mean": round(mean([row[metric] for row in final_rows]), 6),
            "std": round(population_std([row[metric] for row in final_rows]), 6),
        }
        for metric in metric_names
    }
    json_path = output_dir / "final_metrics_mean_std.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    positions = list(range(len(final_rows)))
    labels = [f"seed {row['seed']}" for row in final_rows]
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for offset, metric, color in (
        (-width, "accuracy", "#3182ce"),
        (0, "macro_f1", "#dd6b20"),
        (width, "roc_auc_macro", "#38a169"),
    ):
        axis.bar(
            [position + offset for position in positions],
            [row[metric] for row in final_rows],
            width,
            label=metric,
            color=color,
        )
    axis.set_xticks(positions, labels=labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Final test metrics by seed")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    plot_path = output_dir / "final_seed_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, json_path, plot_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = selected_trials(args.trials, args.search_space)
    validation_seed_pairs = args.validation_seed_pairs or [
        (args.split_seed, model_seed) for model_seed in args.tuning_seeds
    ]
    trial_rows = []

    for config in trials:
        for split_seed, model_seed in validation_seed_pairs:
            run_name = (
                f"tune_{config['name']}_split{split_seed}_seed{model_seed}"
            )
            result_path = output_dir / f"{run_name}.json"
            if not args.resume or not result_path.exists():
                command = training_command(
                    args,
                    config,
                    model_seed,
                    output_dir,
                    run_name,
                    "validation",
                    split_seed=split_seed,
                )
                print("RUN:", " ".join(command), flush=True)
                subprocess.run(command, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = {
                "trial": config["name"],
                "split_seed": split_seed,
                "model_seed": model_seed,
                "best_epoch": result["training"]["best_epoch"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "log_loss": result["log_loss"],
                "expected_calibration_error": result["confidence"][
                    "expected_calibration_error"
                ],
                "temperature": result["calibration"]["temperature"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
            }
            trial_rows.append(row)
            if not args.keep_trial_checkpoints:
                remove_large_trial_files(output_dir, run_name)

    grouped = defaultdict(list)
    for row in trial_rows:
        grouped[row["trial"]].append(row)

    summary_rows = []
    for config in trials:
        rows = grouped[config["name"]]
        f1_values = [row["macro_f1"] for row in rows]
        accuracy_values = [row["accuracy"] for row in rows]
        loss_values = [row["loss"] for row in rows]
        log_loss_values = [row["log_loss"] for row in rows]
        ece_values = [row["expected_calibration_error"] for row in rows]
        mean_f1 = mean(f1_values)
        std_f1 = population_std(f1_values)
        summary_rows.append(
            {
                "trial": config["name"],
                "mean_macro_f1": round(mean_f1, 6),
                "std_macro_f1": round(std_f1, 6),
                "robust_macro_f1": round(mean_f1 - std_f1, 6),
                "mean_accuracy": round(mean(accuracy_values), 6),
                "mean_loss": round(mean(loss_values), 6),
                "mean_log_loss": round(mean(log_loss_values), 6),
                "mean_ece": round(mean(ece_values), 6),
                "validation_seed_pairs": ",".join(
                    f"{split_seed}:{model_seed}"
                    for split_seed, model_seed in validation_seed_pairs
                ),
                **{key: value for key, value in config.items() if key != "name"},
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -row["robust_macro_f1"],
            -row["mean_macro_f1"],
            row["std_macro_f1"],
            row["mean_ece"],
            -row["mean_accuracy"],
            row["mean_log_loss"],
        )
    )
    best_summary = summary_rows[0]
    best_config = next(
        config for config in trials if config["name"] == best_summary["trial"]
    )

    trial_csv = output_dir / "tuning_trials.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as file:
        trial_fields = list(
            dict.fromkeys(key for row in trial_rows for key in row.keys())
        )
        writer = csv.DictWriter(file, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(trial_rows)

    summary_csv = output_dir / "tuning_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        summary_fields = list(
            dict.fromkeys(key for row in summary_rows for key in row.keys())
        )
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    selection = {
        "selection_metric": "mean Macro-F1 minus one standard deviation",
        "search_space": args.search_space,
        "final_split_seed": args.split_seed,
        "validation_seed_pairs": validation_seed_pairs,
        "best_configuration": best_config,
        "best_validation_summary": best_summary,
        "ranking": summary_rows,
    }
    selection_path = output_dir / "selected_configuration.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    save_tuning_plot(summary_rows, output_dir)
    print(json.dumps(selection, indent=2))

    if args.run_final:
        final_output_dir = Path(args.final_output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_rows = []
        for seed in args.final_seeds:
            run_name = f"{args.dataset}_{best_config['name']}_seed{seed}"
            command = training_command(
                args,
                best_config,
                seed,
                final_output_dir,
                run_name,
                "test",
                epochs=args.final_epochs,
            )
            command.append("--refit-full-train")
            print("FINAL RUN:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            result_path = final_output_dir / f"{run_name}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            final_rows.append(
                {
                    "seed": seed,
                    "best_epoch": result["training"]["best_epoch"],
                    "accuracy": result["accuracy"],
                    "precision_macro": result["precision_macro"],
                    "recall_macro": result["recall_macro"],
                    "macro_f1": result["macro_f1"],
                    "weighted_f1": result["weighted_f1"],
                    "roc_auc_macro": result["roc_auc"]["macro"],
                    "log_loss": result["log_loss"],
                    "expected_calibration_error": result["confidence"][
                        "expected_calibration_error"
                    ],
                    "temperature": result["calibration"]["temperature"],
                }
            )
        save_final_summary(final_rows, final_output_dir)


if __name__ == "__main__":
    main()
