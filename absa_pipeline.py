import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PIPELINE_VERSION = "absa-training-v4-clean-ablation"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aspect_samples(path):
    """Expand each sentence into one sample per annotated aspect."""
    with Path(path).open(encoding="utf-8") as file:
        sentences = json.load(file)

    samples = []
    for sentence in sentences:
        sentence_key = " ".join(sentence["sentence"].lower().split())
        for aspect in sentence["aspects"]:
            polarity = aspect["polarity"]
            if polarity not in LABEL_TO_ID:
                continue
            samples.append(
                {
                    "sentence": sentence["sentence"],
                    "sentence_key": sentence_key,
                    "aspect": " ".join(aspect["term"]),
                    "label": LABEL_TO_ID[polarity],
                }
            )
    return samples


def load_aspect_descriptions(data_dir):
    """Load optional aspect descriptions for controlled input-format ablations.

    The default training path still uses only the raw aspect term. Descriptions are
    only used when --input-format term_desc is selected, so the contribution can
    be measured as a separate ablation.
    """
    path = Path(data_dir) / "aspects.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    descriptions = {}
    for aspect, value in raw.items():
        description = value.get("description", "") if isinstance(value, dict) else str(value)
        aspect_key = " ".join(str(aspect).split())
        descriptions[aspect_key] = " ".join(str(description).split())
        descriptions[aspect_key.lower()] = descriptions[aspect_key]
    return descriptions


def remove_train_test_overlap(train_samples, test_samples):
    """Remove complete training sentences that also occur in the official test set."""
    test_sentences = {sample["sentence_key"] for sample in test_samples}
    clean_train = [
        sample for sample in train_samples if sample["sentence_key"] not in test_sentences
    ]
    removed = [
        sample for sample in train_samples if sample["sentence_key"] in test_sentences
    ]
    return clean_train, removed


def grouped_stratified_split(samples, valid_ratio, seed, candidates=256):
    """Choose a sentence-disjoint split with label counts close to stratified targets."""
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")

    labels = np.asarray([sample["label"] for sample in samples])
    groups = np.asarray([sample["sentence_key"] for sample in samples])
    target_size = len(samples) * valid_ratio
    total_counts = np.bincount(labels, minlength=len(LABELS))
    target_counts = total_counts * valid_ratio
    unique_groups = sorted(set(groups))
    valid_group_count = max(1, round(len(unique_groups) * valid_ratio))
    rng = random.Random(seed)
    best = None
    for _ in range(candidates):
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        valid_groups = set(shuffled_groups[:valid_group_count])
        valid_indices = np.flatnonzero(np.isin(groups, list(valid_groups)))
        train_indices = np.flatnonzero(~np.isin(groups, list(valid_groups)))
        valid_counts = np.bincount(labels[valid_indices], minlength=len(LABELS))
        size_error = abs(len(valid_indices) - target_size) / max(target_size, 1)
        label_error = np.mean(
            np.abs(valid_counts - target_counts) / np.maximum(target_counts, 1)
        )
        score = size_error + label_error
        if best is None or score < best[0]:
            best = (score, train_indices, valid_indices)

    _, train_indices, valid_indices = best
    train = [samples[index] for index in train_indices]
    valid = [samples[index] for index in valid_indices]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)

    train_sentences = {sample["sentence_key"] for sample in train}
    valid_sentences = {sample["sentence_key"] for sample in valid}
    if train_sentences & valid_sentences:
        raise RuntimeError("Sentence leakage detected between train and validation")
    return train, valid


def assert_no_sentence_overlap(left, right, names):
    overlap = (
        {sample["sentence_key"] for sample in left}
        & {sample["sentence_key"] for sample in right}
    )
    if overlap:
        raise RuntimeError(
            f"Sentence leakage detected between {names[0]} and {names[1]}: "
            f"{len(overlap)} shared sentence(s)"
        )


class ABSADataset(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        max_length,
        seen_aspects,
        input_format="term",
        aspect_descriptions=None,
    ):
        if input_format not in {"term", "term_desc"}:
            raise ValueError(f"Unsupported input_format: {input_format}")
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seen_aspects = seen_aspects
        self.input_format = input_format
        self.aspect_descriptions = aspect_descriptions or {}

    def __len__(self):
        return len(self.samples)

    def _aspect_pair_text_and_span(self, aspect):
        """Return text_pair plus the char span of the actual aspect term.

        For term_desc, aspect_mask intentionally covers only the aspect term, not
        the generated description. This keeps aspect_attention/hybrid comparable
        to the term-only setup and avoids diluting the aspect query with a long
        paraphrase.
        """
        if self.input_format == "term":
            return aspect, 0, len(aspect)

        description = self.aspect_descriptions.get(
            aspect, self.aspect_descriptions.get(aspect.lower(), "")
        )
        prefix = "aspect: "
        if description:
            pair_text = f"{prefix}{aspect}. description: {description}"
        else:
            pair_text = f"{prefix}{aspect}"
        return pair_text, len(prefix), len(prefix) + len(aspect)

    def __getitem__(self, index):
        sample = self.samples[index]
        aspect = sample["aspect"]
        aspect_pair_text, aspect_start, aspect_end = self._aspect_pair_text_and_span(aspect)
        encoded = self.tokenizer(
            sample["sentence"],
            aspect_pair_text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        sequence_ids = encoded.sequence_ids()
        sentence_mask = [sequence_id == 0 for sequence_id in sequence_ids]
        aspect_mask = []
        for sequence_id, (start, end) in zip(sequence_ids, offsets):
            keep = (
                sequence_id == 1
                and end > aspect_start
                and start < aspect_end
                and end > start
            )
            aspect_mask.append(bool(keep))
        # Very defensive fallback for unusual tokenizers/offset mappings.
        if not any(aspect_mask):
            aspect_mask = [sequence_id == 1 for sequence_id in sequence_ids]
        encoded["sentence_mask"] = sentence_mask
        encoded["aspect_mask"] = aspect_mask
        encoded["labels"] = sample["label"]
        encoded["is_seen"] = int(aspect in self.seen_aspects)
        return encoded


@dataclass
class BatchCollator:
    tokenizer: object

    def __call__(self, rows):
        model_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"labels", "is_seen", "sentence_mask", "aspect_mask"}
            }
            for row in rows
        ]
        batch = self.tokenizer.pad(model_rows, padding=True, return_tensors="pt")

        def pad_mask(key):
            padded = []
            max_len = max(len(row[key]) for row in rows)
            for row in rows:
                values = list(row[key])
                padded.append(values + [False] * (max_len - len(values)))
            return torch.tensor(padded, dtype=torch.bool)

        batch["sentence_mask"] = pad_mask("sentence_mask")
        batch["aspect_mask"] = pad_mask("aspect_mask")
        batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        batch["is_seen"] = torch.tensor([row["is_seen"] for row in rows], dtype=torch.bool)
        return batch


class ABSAClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        dropout=0.2,
        class_weights=None,
        label_smoothing=0.0,
        pooling_strategy="cls",
        encoder_layer_pooling="last",
        loss_type="cross_entropy",
        focal_gamma=1.5,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.label_smoothing = label_smoothing
        self.pooling_strategy = pooling_strategy
        self.encoder_layer_pooling = encoder_layer_pooling
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        if class_weights is None:
            class_weights = torch.ones(len(LABELS), dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        if pooling_strategy not in {"cls", "aspect_attention", "hybrid"}:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        if encoder_layer_pooling not in {"last", "mean_last4"}:
            raise ValueError(
                f"Unsupported encoder layer pooling: {encoder_layer_pooling}"
            )
        if loss_type not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        if pooling_strategy != "cls":
            self.aspect_query = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            input_size = hidden_size * (3 if pooling_strategy == "hybrid" else 2)
            self.representation = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(LABELS)),
        )

    @staticmethod
    def masked_mean(hidden, mask):
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        aspect_mask,
        sentence_mask,
        labels=None,
        **encoder_inputs,
    ):
        outputs = self.encoder(
            **encoder_inputs,
            output_hidden_states=self.encoder_layer_pooling == "mean_last4",
        )
        if self.encoder_layer_pooling == "mean_last4":
            hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
        else:
            hidden = outputs.last_hidden_state
        cls_repr = hidden[:, 0]
        if self.pooling_strategy == "cls":
            representation = cls_repr
        else:
            aspect_repr = self.masked_mean(hidden, aspect_mask)
            query = self.aspect_query(aspect_repr).unsqueeze(-1)
            attention_scores = torch.bmm(hidden, query).squeeze(-1) / math.sqrt(
                hidden.shape[-1]
            )
            attention_scores = attention_scores.masked_fill(~sentence_mask, -1e4)
            attention = torch.softmax(attention_scores, dim=-1)
            context_repr = torch.bmm(attention.unsqueeze(1), hidden).squeeze(1)
            features = [context_repr, aspect_repr]
            if self.pooling_strategy == "hybrid":
                features.insert(0, cls_repr)
            representation = self.representation(torch.cat(features, dim=-1))
        logits = self.classifier(representation)

        loss = None
        if labels is not None:
            per_sample_loss = nn.functional.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            if self.loss_type == "focal":
                true_class_probability = torch.softmax(logits.float(), dim=-1).gather(
                    1, labels.unsqueeze(1)
                ).squeeze(1)
                per_sample_loss = (
                    (1.0 - true_class_probability).pow(self.focal_gamma)
                    * per_sample_loss
                )
            loss = per_sample_loss.mean()
        return loss, logits


def build_loaders(
    data_dir,
    model_name,
    batch_size,
    max_length,
    valid_ratio,
    split_seed,
    input_format="term",
):
    data_dir = Path(data_dir)
    train_raw = load_aspect_samples(data_dir / "train.multiple.json")
    test = load_aspect_samples(data_dir / "test.multiple.json")
    aspect_descriptions = load_aspect_descriptions(data_dir)
    train_all, excluded_overlap = remove_train_test_overlap(train_raw, test)
    train, valid = grouped_stratified_split(train_all, valid_ratio, split_seed)
    assert_no_sentence_overlap(train, valid, ("train", "validation"))
    assert_no_sentence_overlap(train_all, test, ("train", "test"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = BatchCollator(tokenizer)
    seen_aspects = {sample["aspect"] for sample in train}

    def loader(samples, shuffle):
        dataset = ABSADataset(
            samples,
            tokenizer,
            max_length,
            seen_aspects,
            input_format=input_format,
            aspect_descriptions=aspect_descriptions,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    return loader(train, True), loader(valid, False), loader(test, False), {
        "train": train,
        "train_all": train_all,
        "valid": valid,
        "test": test,
        "excluded_train_test_overlap": excluded_overlap,
    }


def build_full_train_loaders(
    data_dir,
    model_name,
    batch_size,
    max_length,
    input_format="term",
):
    data_dir = Path(data_dir)
    train_raw = load_aspect_samples(data_dir / "train.multiple.json")
    test = load_aspect_samples(data_dir / "test.multiple.json")
    aspect_descriptions = load_aspect_descriptions(data_dir)
    train, excluded_overlap = remove_train_test_overlap(train_raw, test)
    assert_no_sentence_overlap(train, test, ("full train", "test"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = BatchCollator(tokenizer)
    seen_aspects = {sample["aspect"] for sample in train}

    def loader(samples, shuffle):
        dataset = ABSADataset(
            samples,
            tokenizer,
            max_length,
            seen_aspects,
            input_format=input_format,
            aspect_descriptions=aspect_descriptions,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    return loader(train, True), loader(test, False), {
        "train": train,
        "test": test,
        "excluded_train_test_overlap": excluded_overlap,
    }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def encoder_inputs_from_batch(batch):
    supported_keys = {"input_ids", "attention_mask", "token_type_ids"}
    return {key: value for key, value in batch.items() if key in supported_keys}


def _probabilities_and_predictions(
    model,
    loader,
    device,
    use_amp=False,
    temperature=1.0,
):
    model.eval()
    losses, predictions, labels, seen_flags, confidences = [], [], [], [], []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                loss, logits = model(
                    **encoder_inputs_from_batch(batch),
                    labels=batch["labels"],
                    aspect_mask=batch["aspect_mask"],
                    sentence_mask=batch["sentence_mask"],
                )
            losses.append(loss.item())
            # AMP can leave logits in float16. Compute probabilities in float32
            # so downstream probability metrics receive stable values.
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            probabilities.extend(probs.cpu().tolist())
            predictions.extend(probs.argmax(dim=-1).cpu().tolist())
            confidences.extend(probs.max(dim=-1).values.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            seen_flags.extend(batch["is_seen"].cpu().tolist())
    return losses, predictions, labels, seen_flags, confidences, probabilities


def evaluate(
    model,
    loader,
    device,
    use_amp=False,
    include_predictions=False,
    temperature=1.0,
):
    losses, predictions, labels, seen_flags, confidences, probabilities = _probabilities_and_predictions(
        model,
        loader,
        device,
        use_amp=use_amp,
        temperature=temperature,
    )

    def subset_metrics(mask):
        selected_labels = [label for label, keep in zip(labels, mask) if keep]
        selected_predictions = [
            prediction for prediction, keep in zip(predictions, mask) if keep
        ]
        if not selected_labels:
            return {"samples": 0, "accuracy": None, "macro_f1": None}
        return {
            "samples": len(selected_labels),
            "accuracy": round(accuracy_score(selected_labels, selected_predictions), 4),
            "macro_f1": round(f1_score(selected_labels, selected_predictions, average="macro"), 4),
        }

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=range(len(LABELS)),
        zero_division=0,
    )
    aggregate_metrics = {}
    for average in ("macro", "micro", "weighted"):
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average=average,
            zero_division=0,
        )
        aggregate_metrics[average] = {
            "precision": round(float(avg_precision), 4),
            "recall": round(float(avg_recall), 4),
            "f1": round(float(avg_f1), 4),
        }

    one_hot = label_binarize(labels, classes=range(len(LABELS)))
    prob_matrix = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(prob_matrix).all():
        raise ValueError("Model probabilities contain NaN or infinite values")
    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Model probabilities contain a row with zero total mass")
    prob_matrix = prob_matrix / row_sums
    probabilities = prob_matrix.tolist()
    predictions = prob_matrix.argmax(axis=1).tolist()
    confidences = prob_matrix.max(axis=1).tolist()
    auc_scores = {"macro": None, "micro": None, "weighted": None}
    auc_per_class = {}
    if len(np.unique(labels)) > 1:
        try:
            for average in auc_scores:
                if average == "micro":
                    score = roc_auc_score(one_hot.ravel(), prob_matrix.ravel())
                else:
                    score = roc_auc_score(
                        one_hot,
                        prob_matrix,
                        average=average,
                        multi_class="ovr",
                    )
                auc_scores[average] = round(float(score), 4)
            class_aucs = roc_auc_score(one_hot, prob_matrix, average=None, multi_class="ovr")
            auc_per_class = {
                LABELS[index]: round(float(score), 4) for index, score in enumerate(class_aucs)
            }
        except ValueError:
            auc_scores = {"macro": None, "micro": None, "weighted": None}
            auc_per_class = {}

    correct_mask = [int(p == y) for p, y in zip(predictions, labels)]
    wrong_mask = [not bool(flag) for flag in correct_mask]
    avg_confidence = round(float(np.mean(confidences)), 4) if confidences else None
    avg_confidence_correct = (
        round(float(np.mean([c for c, ok in zip(confidences, correct_mask) if ok])), 4)
        if any(correct_mask)
        else None
    )
    avg_confidence_wrong = (
        round(float(np.mean([c for c, ok in zip(confidences, wrong_mask) if ok])), 4)
        if any(wrong_mask)
        else None
    )
    bin_edges = np.linspace(0.0, 1.0, 11)
    calibration_bins = []
    expected_calibration_error = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (np.asarray(confidences) >= lower) & (np.asarray(confidences) <= upper)
        else:
            mask = (np.asarray(confidences) >= lower) & (np.asarray(confidences) < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(np.asarray(confidences)[mask].mean())
        bin_accuracy = float(np.asarray(correct_mask)[mask].mean())
        expected_calibration_error += (count / len(labels)) * abs(
            bin_accuracy - bin_confidence
        )
        calibration_bins.append(
            {
                "lower": round(float(lower), 2),
                "upper": round(float(upper), 2),
                "count": count,
                "accuracy": round(bin_accuracy, 4),
                "confidence": round(bin_confidence, 4),
            }
        )
    brier_score = float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))

    roc_curve_data = {}
    if len(np.unique(labels)) > 1:
        for index, label in enumerate(LABELS):
            try:
                fpr, tpr, _ = roc_curve(one_hot[:, index], prob_matrix[:, index])
                roc_curve_data[label] = {
                    "fpr": fpr.round(6).tolist(),
                    "tpr": tpr.round(6).tolist(),
                    "auc": auc_per_class.get(label),
                }
            except ValueError:
                continue

    result = {
        "samples": len(labels),
        "loss": round(float(np.mean(losses)), 4) if losses else None,
        "log_loss": round(float(log_loss(labels, prob_matrix, labels=range(len(LABELS)))), 4),
        "accuracy": round(accuracy_score(labels, predictions), 4) if labels else None,
        "balanced_accuracy": round(balanced_accuracy_score(labels, predictions), 4),
        "precision_macro": aggregate_metrics["macro"]["precision"],
        "recall_macro": aggregate_metrics["macro"]["recall"],
        "macro_f1": aggregate_metrics["macro"]["f1"],
        "precision_micro": aggregate_metrics["micro"]["precision"],
        "recall_micro": aggregate_metrics["micro"]["recall"],
        "micro_f1": aggregate_metrics["micro"]["f1"],
        "precision_weighted": aggregate_metrics["weighted"]["precision"],
        "recall_weighted": aggregate_metrics["weighted"]["recall"],
        "weighted_f1": aggregate_metrics["weighted"]["f1"],
        "aggregate_metrics": aggregate_metrics,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=range(len(LABELS))
        ).tolist(),
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(per_class_f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "roc_auc": {**auc_scores, "per_class": auc_per_class},
        "confidence": {
            "average": avg_confidence,
            "average_correct": avg_confidence_correct,
            "average_incorrect": avg_confidence_wrong,
            "expected_calibration_error": round(expected_calibration_error, 4),
            "brier_score": round(brier_score, 4),
            "calibration_bins": calibration_bins,
        },
        "seen": subset_metrics(seen_flags),
        "unseen": subset_metrics([not flag for flag in seen_flags]),
        "roc_curve": roc_curve_data,
    }
    if include_predictions:
        result["prediction_rows"] = [
            {
                "gold_label": LABELS[label],
                "predicted_label": LABELS[prediction],
                "confidence": round(float(confidence), 6),
                **{
                    f"probability_{class_name}": round(float(probability), 6)
                    for class_name, probability in zip(LABELS, sample_probabilities)
                },
                "is_seen": bool(is_seen),
                "correct": label == prediction,
            }
            for label, prediction, confidence, is_seen, sample_probabilities in zip(
                labels, predictions, confidences, seen_flags, probabilities
            )
        ]
    return result


def collect_logits(model, loader, device, use_amp=False):
    model.eval()
    logits_rows = []
    label_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                _, logits = model(
                    **encoder_inputs_from_batch(batch),
                    labels=None,
                    aspect_mask=batch["aspect_mask"],
                    sentence_mask=batch["sentence_mask"],
                )
            logits_rows.append(logits.detach().float().cpu())
            label_rows.append(batch["labels"].detach().cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)


def fit_temperature(model, loader, device, use_amp=False, max_iter=50):
    logits, labels = collect_logits(model, loader, device, use_amp=use_amp)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0))


def build_optimizer(
    model,
    encoder_learning_rate=1e-5,
    head_learning_rate=1e-5,
    weight_decay=0.01,
):
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = []
    for is_encoder, learning_rate in (
        (True, encoder_learning_rate),
        (False, head_learning_rate),
    ):
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("encoder.") == is_encoder
        ]
        for apply_decay in (True, False):
            parameters = [
                parameter
                for name, parameter in named_parameters
                if (not any(term in name for term in no_decay_terms)) == apply_decay
            ]
            if parameters:
                parameter_groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay if apply_decay else 0.0,
                    }
                )
    return torch.optim.AdamW(parameter_groups)


def build_scheduler(optimizer, total_steps, warmup_ratio, scheduler_type):
    if scheduler_type == "none":
        return None
    if scheduler_type != "linear":
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )


def _train_loop(
    model,
    train_loader,
    valid_loader,
    device,
    epochs,
    output_path,
    patience=4,
    gradient_accumulation_steps=1,
    warmup_ratio=0.0,
    use_amp=False,
    optimizer_builder=None,
    encoder_learning_rate=1e-5,
    head_learning_rate=1e-5,
    weight_decay=0.01,
    scheduler_type="none",
    early_stopping_min_delta=0.0,
):
    optimizer = (
        optimizer_builder(model)
        if optimizer_builder is not None
        else build_optimizer(
            model,
            encoder_learning_rate=encoder_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=weight_decay,
        )
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio, scheduler_type)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                loss, _ = model(
                    **encoder_inputs_from_batch(batch),
                    labels=batch["labels"],
                    aspect_mask=batch["aspect_mask"],
                    sentence_mask=batch["sentence_mask"],
                )
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())

        metrics = evaluate(model, valid_loader, device, use_amp=use_amp)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        eta_seconds = epoch_seconds * (epochs - epoch)
        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "valid_loss": metrics["loss"],
            "valid_accuracy": metrics["accuracy"],
            "valid_macro_f1": metrics["macro_f1"],
            "valid_weighted_f1": metrics["weighted_f1"],
            "valid_auc_macro": metrics["roc_auc"]["macro"],
            "valid_avg_confidence": metrics["confidence"]["average"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 2),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "eta_seconds": round(eta_seconds, 2),
        }
        history.append(record)
        print(
            f"epoch={epoch}/{epochs} train_loss={record['train_loss']:.4f} "
            f"valid_loss={metrics['loss']:.4f} valid_acc={metrics['accuracy']:.4f} "
            f"valid_f1={metrics['macro_f1']:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={epoch_seconds:.1f}s eta={eta_seconds / 60:.1f}m"
        )
        if metrics["macro_f1"] > best_f1 + early_stopping_min_delta:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), output_path)
            print(f"saved best checkpoint: epoch={epoch} valid_f1={best_f1:.4f}")
        else:
            stale_epochs += 1
            print(f"no improvement: {stale_epochs}/{patience}")
            if patience > 0 and stale_epochs >= patience:
                print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    summary = {
        "best_epoch": best_epoch,
        "best_valid_macro_f1": round(best_f1, 4),
        "total_seconds": round(time.perf_counter() - training_start, 2),
        "history": history,
    }
    return model, summary


def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    epochs,
    output_path,
    patience=4,
    gradient_accumulation_steps=1,
    warmup_ratio=0.0,
    use_amp=False,
    optimizer_builder=None,
    encoder_learning_rate=1e-5,
    head_learning_rate=1e-5,
    weight_decay=0.01,
    scheduler_type="none",
    early_stopping_min_delta=0.0,
):
    return _train_loop(
        model,
        train_loader,
        valid_loader,
        device,
        epochs,
        output_path,
        patience=patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        use_amp=use_amp,
        optimizer_builder=optimizer_builder,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
        early_stopping_min_delta=early_stopping_min_delta,
    )


def train_fixed_epochs(
    model,
    train_loader,
    device,
    epochs,
    output_path,
    gradient_accumulation_steps=1,
    warmup_ratio=0.0,
    use_amp=False,
    optimizer_builder=None,
    encoder_learning_rate=1e-5,
    head_learning_rate=1e-5,
    weight_decay=0.01,
    scheduler_type="none",
    scheduler_total_epochs=None,
):
    optimizer = (
        optimizer_builder(model)
        if optimizer_builder is not None
        else build_optimizer(
            model,
            encoder_learning_rate=encoder_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=weight_decay,
        )
    )
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    schedule_epochs = scheduler_total_epochs or epochs
    if schedule_epochs < epochs:
        raise ValueError("scheduler_total_epochs cannot be smaller than training epochs")
    total_steps = steps_per_epoch * schedule_epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio, scheduler_type)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                loss, _ = model(
                    **encoder_inputs_from_batch(batch),
                    labels=batch["labels"],
                    aspect_mask=batch["aspect_mask"],
                    sentence_mask=batch["sentence_mask"],
                )
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 2),
        }
        history.append(record)
        print(
            f"refit_epoch={epoch}/{epochs} train_loss={record['train_loss']:.4f} "
            f"lr={record['learning_rate']:.2e} time={epoch_seconds:.1f}s"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return model, {
        "epochs": epochs,
        "total_seconds": round(time.perf_counter() - training_start, 2),
        "history": history,
    }


def describe_samples(splits):
    result = {}
    train_aspects = {sample["aspect"] for sample in splits.get("train", [])}
    for name, samples in splits.items():
        counts = Counter(LABELS[sample["label"]] for sample in samples)
        aspects = {sample["aspect"] for sample in samples}
        summary = {
            "samples": len(samples),
            "labels": dict(sorted(counts.items())),
            "unique_aspects": len(aspects),
        }
        if name != "train" and train_aspects:
            unseen = [sample for sample in samples if sample["aspect"] not in train_aspects]
            summary["unseen_aspect_samples"] = len(unseen)
            summary["unseen_aspect_ratio"] = round(len(unseen) / max(len(samples), 1), 4)
        result[name] = summary
    return result
