#!/usr/bin/env python3
"""
Step 3: Python deep learning comparison for Dataset 3.

This script trains five neural-network architectures on the same Dataset 3
classification task used in the R machine-learning stage.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_VERSION = "research_layer_step03_five_dl_methods_v1"

SCRIPT_FOLDER = Path(__file__).resolve().parent
TARGET_COLUMN = "medium_or_high_influence"
ID_COLUMN = "canonical_entity_id"

RANDOM_SEED = 42
TRAIN_SHARE = 0.75
THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 5)]

LEAKAGE_COLUMNS = {
    "rank_overall",
    "influence_score",
    "influence_band",
    "influence_band_clean",
}


@dataclass
class TrainingExample:
    entity_id: str
    features: list[float]
    target: int


@dataclass
class ScalingStats:
    feature_names: list[str]
    means: list[float]
    standard_deviations: list[float]


@dataclass
class TrainTestSplit:
    train_examples: list[TrainingExample]
    test_examples: list[TrainingExample]


@dataclass
class ModelConfig:
    model_name: str
    hidden_layers: list[int]
    learning_rate: float
    epoch_count: int
    dropout_rate: float


@dataclass
class NeuralNetwork:
    hidden_weights: list[list[list[float]]]
    hidden_biases: list[list[float]]
    output_weights: list[float]
    output_bias: float


def input_candidates() -> list[Path]:
    return [
        SCRIPT_FOLDER.parent / "outputs" / "dataset3_model_feature_table.csv",
        SCRIPT_FOLDER / "outputs" / "dataset3_model_feature_table.csv",
        Path.cwd() / "research_layer" / "outputs" / "dataset3_model_feature_table.csv",
        Path.cwd() / "outputs" / "dataset3_model_feature_table.csv",
        Path.cwd() / "dataset3_model_feature_table.csv",
    ]


def find_input_file() -> Path:
    for file_path in input_candidates():
        if file_path.exists():
            return file_path
    raise FileNotFoundError("Could not find dataset3_model_feature_table.csv.")


def clean_text(raw_value: object) -> str:
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def safe_float(raw_value: object) -> float:
    try:
        return float(clean_text(raw_value).replace(",", ""))
    except ValueError:
        return 0.0


def safe_int(raw_value: object) -> int:
    try:
        return int(float(clean_text(raw_value)))
    except ValueError:
        return 0


def load_feature_rows(file_path: Path) -> list[dict[str, str]]:
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except FileNotFoundError as read_error:
        raise FileNotFoundError(f"Missing input file: {file_path}") from read_error


def validate_feature_rows(feature_rows: list[dict[str, str]]) -> None:
    if not feature_rows:
        raise ValueError("The feature table is empty.")
    if TARGET_COLUMN not in feature_rows[0]:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")


def candidate_feature_names(feature_rows: list[dict[str, str]]) -> list[str]:
    blocked_columns = {TARGET_COLUMN, ID_COLUMN}.union(LEAKAGE_COLUMNS)
    return [column_name for column_name in feature_rows[0] if column_name not in blocked_columns]


def column_has_variation(feature_rows: list[dict[str, str]], column_name: str) -> bool:
    values = {safe_float(feature_row.get(column_name, 0)) for feature_row in feature_rows}
    return len(values) > 1


def selected_feature_names(feature_rows: list[dict[str, str]]) -> list[str]:
    candidate_names = candidate_feature_names(feature_rows)
    return [name for name in candidate_names if column_has_variation(feature_rows, name)]


def build_training_example(row_data: dict[str, str], feature_names: list[str]) -> TrainingExample:
    entity_id = clean_text(row_data.get(ID_COLUMN, "unknown"))
    features = [safe_float(row_data.get(feature_name, 0)) for feature_name in feature_names]
    target = safe_int(row_data.get(TARGET_COLUMN, 0))
    return TrainingExample(entity_id=entity_id, features=features, target=target)


def validate_examples(examples: list[TrainingExample], feature_names: list[str]) -> None:
    if not feature_names:
        raise ValueError("No useful feature columns remain.")
    if len({example.target for example in examples}) < 2:
        raise ValueError("The target needs both positive and negative classes.")


def prepare_training_examples(feature_rows: list[dict[str, str]]) -> tuple[list[TrainingExample], list[str]]:
    validate_feature_rows(feature_rows)
    feature_names = selected_feature_names(feature_rows)
    examples = [build_training_example(row_data, feature_names) for row_data in feature_rows]
    validate_examples(examples, feature_names)
    return examples, feature_names


def class_indices(examples: list[TrainingExample], target_value: int) -> list[int]:
    return [index for index, example in enumerate(examples) if example.target == target_value]


def sample_indices(indices: list[int], train_share: float) -> list[int]:
    sample_count = max(1, int(len(indices) * train_share))
    return random.sample(indices, min(sample_count, len(indices)))


def make_stratified_split(examples: list[TrainingExample]) -> TrainTestSplit:
    positive_indices = class_indices(examples, 1)
    negative_indices = class_indices(examples, 0)
    train_indices = set(sample_indices(positive_indices, TRAIN_SHARE))
    train_indices.update(sample_indices(negative_indices, TRAIN_SHARE))
    return build_split_from_indices(examples, train_indices)


def build_split_from_indices(examples: list[TrainingExample], train_indices: set[int]) -> TrainTestSplit:
    train_examples = [example for index, example in enumerate(examples) if index in train_indices]
    test_examples = [example for index, example in enumerate(examples) if index not in train_indices]
    return TrainTestSplit(train_examples=train_examples, test_examples=test_examples)


def validate_split(split_data: TrainTestSplit) -> None:
    if not split_data.train_examples or not split_data.test_examples:
        raise ValueError("Train/test split failed because one split is empty.")


def feature_mean(examples: list[TrainingExample], feature_index: int) -> float:
    values = [example.features[feature_index] for example in examples]
    return sum(values) / len(values)


def feature_standard_deviation(examples: list[TrainingExample], feature_index: int, mean_value: float) -> float:
    values = [example.features[feature_index] for example in examples]
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance) or 1.0


def fit_scaling_stats(examples: list[TrainingExample], feature_names: list[str]) -> ScalingStats:
    means = [feature_mean(examples, index) for index in range(len(feature_names))]
    deviations = [feature_standard_deviation(examples, index, means[index]) for index in range(len(feature_names))]
    return ScalingStats(feature_names=feature_names, means=means, standard_deviations=deviations)


def scale_feature_values(features: list[float], scaling_stats: ScalingStats) -> list[float]:
    scaled_values = []
    for index, feature_value in enumerate(features):
        centred_value = feature_value - scaling_stats.means[index]
        scaled_values.append(centred_value / scaling_stats.standard_deviations[index])
    return scaled_values


def scale_example(example: TrainingExample, scaling_stats: ScalingStats) -> TrainingExample:
    scaled_features = scale_feature_values(example.features, scaling_stats)
    return TrainingExample(example.entity_id, scaled_features, example.target)


def scale_examples(examples: list[TrainingExample], scaling_stats: ScalingStats) -> list[TrainingExample]:
    return [scale_example(example, scaling_stats) for example in examples]


def model_configs() -> list[ModelConfig]:
    return [
        ModelConfig("shallow_neural_network", [8], 0.003, 80, 0.0),
        ModelConfig("deep_neural_network", [16, 8], 0.003, 90, 0.0),
        ModelConfig("wide_neural_network", [32], 0.0025, 80, 0.0),
        ModelConfig("narrow_deep_neural_network", [12, 8, 4], 0.003, 90, 0.0),
        ModelConfig("dropout_neural_network", [16, 8], 0.003, 100, 0.20),
    ]


def random_weight() -> float:
    return random.uniform(-0.1, 0.1)


def layer_sizes(input_count: int, config: ModelConfig) -> list[int]:
    return [input_count] + config.hidden_layers


def initialise_layer(input_count: int, output_count: int) -> list[list[float]]:
    return [[random_weight() for _ in range(output_count)] for _ in range(input_count)]


def initialise_network(input_count: int, config: ModelConfig) -> NeuralNetwork:
    sizes = layer_sizes(input_count, config)
    hidden_weights = [initialise_layer(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)]
    hidden_biases = [[0.0 for _ in range(size)] for size in config.hidden_layers]
    output_weights = [random_weight() for _ in range(config.hidden_layers[-1])]
    return NeuralNetwork(hidden_weights, hidden_biases, output_weights, 0.0)


def stable_sigmoid(raw_value: float) -> float:
    if raw_value > 35:
        return 1.0
    if raw_value < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-raw_value))


def hidden_activation(raw_value: float) -> float:
    return math.tanh(raw_value)


def layer_output(inputs: list[float], weights: list[list[float]], biases: list[float]) -> list[float]:
    outputs = []
    for output_index, bias_value in enumerate(biases):
        raw_value = bias_value + weighted_input_sum(inputs, weights, output_index)
        outputs.append(hidden_activation(raw_value))
    return outputs


def weighted_input_sum(inputs: list[float], weights: list[list[float]], output_index: int) -> float:
    return sum(input_value * weights[input_index][output_index] for input_index, input_value in enumerate(inputs))


def dropout_mask(values: list[float], dropout_rate: float, training_mode: bool) -> list[int]:
    if not training_mode or dropout_rate <= 0:
        return [1 for _ in values]
    return [0 if random.random() < dropout_rate else 1 for _ in values]


def apply_dropout(values: list[float], mask: list[int]) -> list[float]:
    return [value * mask[index] for index, value in enumerate(values)]


def forward_hidden_layers(network: NeuralNetwork, features: list[float], config: ModelConfig, training_mode: bool) -> tuple[list[list[float]], list[list[int]]]:
    hidden_layers = []
    masks = []
    current_values = features
    for layer_index, layer_weights in enumerate(network.hidden_weights):
        current_values = layer_output(current_values, layer_weights, network.hidden_biases[layer_index])
        mask = dropout_mask(current_values, config.dropout_rate, training_mode)
        current_values = apply_dropout(current_values, mask)
        hidden_layers.append(current_values)
        masks.append(mask)
    return hidden_layers, masks


def output_score(network: NeuralNetwork, hidden_values: list[float]) -> float:
    output_logit = network.output_bias
    output_logit += sum(value * network.output_weights[index] for index, value in enumerate(hidden_values))
    return stable_sigmoid(output_logit)


def forward_pass(network: NeuralNetwork, features: list[float], config: ModelConfig, training_mode: bool) -> tuple[list[list[float]], list[list[int]], float]:
    hidden_layers, masks = forward_hidden_layers(network, features, config, training_mode)
    score = output_score(network, hidden_layers[-1])
    return hidden_layers, masks, score


def positive_class_weight(examples: list[TrainingExample]) -> float:
    positive_count = sum(example.target == 1 for example in examples)
    negative_count = sum(example.target == 0 for example in examples)
    return negative_count / positive_count if positive_count else 1.0


def class_weight(example: TrainingExample, positive_weight: float) -> float:
    return positive_weight if example.target == 1 else 1.0


def clipped_error(error_value: float) -> float:
    return max(-5.0, min(5.0, error_value))


def weighted_binary_loss(target_value: int, score_value: float, weight_value: float) -> float:
    safe_score = min(0.999999, max(0.000001, score_value))
    positive_loss = -target_value * math.log(safe_score)
    negative_loss = -(1 - target_value) * math.log(1 - safe_score)
    return (positive_loss + negative_loss) * weight_value


def copy_hidden_weights(network: NeuralNetwork) -> list[list[list[float]]]:
    return [[list(row) for row in layer] for layer in network.hidden_weights]


def final_hidden_delta(output_error: float, output_weights: list[float], hidden_values: list[float], mask: list[int]) -> list[float]:
    deltas = []
    for index, hidden_value in enumerate(hidden_values):
        gradient = 1 - hidden_value ** 2
        deltas.append(output_error * output_weights[index] * gradient * mask[index])
    return deltas


def previous_hidden_delta(next_delta: list[float], next_weights: list[list[float]], hidden_values: list[float], mask: list[int]) -> list[float]:
    deltas = []
    for hidden_index, hidden_value in enumerate(hidden_values):
        downstream = sum(next_delta[next_index] * next_weights[hidden_index][next_index] for next_index in range(len(next_delta)))
        deltas.append(downstream * (1 - hidden_value ** 2) * mask[hidden_index])
    return deltas


def build_hidden_deltas(output_error: float, old_output_weights: list[float], old_hidden_weights: list[list[list[float]]], hidden_layers: list[list[float]], masks: list[list[int]]) -> list[list[float]]:
    deltas = [final_hidden_delta(output_error, old_output_weights, hidden_layers[-1], masks[-1])]
    for layer_index in range(len(hidden_layers) - 2, -1, -1):
        next_weights = old_hidden_weights[layer_index + 1]
        deltas.insert(0, previous_hidden_delta(deltas[0], next_weights, hidden_layers[layer_index], masks[layer_index]))
    return deltas


def update_output_layer(network: NeuralNetwork, hidden_values: list[float], output_error: float, learning_rate: float) -> None:
    for index, hidden_value in enumerate(hidden_values):
        network.output_weights[index] -= learning_rate * output_error * hidden_value
    network.output_bias -= learning_rate * output_error


def layer_inputs(features: list[float], hidden_layers: list[list[float]], layer_index: int) -> list[float]:
    if layer_index == 0:
        return features
    return hidden_layers[layer_index - 1]


def update_hidden_layer(network: NeuralNetwork, input_values: list[float], layer_index: int, deltas: list[float], learning_rate: float) -> None:
    for input_index, input_value in enumerate(input_values):
        for output_index, delta_value in enumerate(deltas):
            network.hidden_weights[layer_index][input_index][output_index] -= learning_rate * delta_value * input_value
    update_hidden_biases(network, layer_index, deltas, learning_rate)


def update_hidden_biases(network: NeuralNetwork, layer_index: int, deltas: list[float], learning_rate: float) -> None:
    for output_index, delta_value in enumerate(deltas):
        network.hidden_biases[layer_index][output_index] -= learning_rate * delta_value


def update_all_hidden_layers(network: NeuralNetwork, features: list[float], hidden_layers: list[list[float]], hidden_deltas: list[list[float]], learning_rate: float) -> None:
    for layer_index, deltas in enumerate(hidden_deltas):
        input_values = layer_inputs(features, hidden_layers, layer_index)
        update_hidden_layer(network, input_values, layer_index, deltas, learning_rate)


def train_one_example(network: NeuralNetwork, example: TrainingExample, config: ModelConfig, positive_weight: float) -> float:
    hidden_layers, masks, score = forward_pass(network, example.features, config, True)
    weighted_error = clipped_error((score - example.target) * class_weight(example, positive_weight))
    old_output_weights = list(network.output_weights)
    old_hidden_weights = copy_hidden_weights(network)
    hidden_deltas = build_hidden_deltas(weighted_error, old_output_weights, old_hidden_weights, hidden_layers, masks)
    update_output_layer(network, hidden_layers[-1], weighted_error, config.learning_rate)
    update_all_hidden_layers(network, example.features, hidden_layers, hidden_deltas, config.learning_rate)
    return weighted_binary_loss(example.target, score, class_weight(example, positive_weight))


def train_epoch(network: NeuralNetwork, examples: list[TrainingExample], config: ModelConfig, positive_weight: float) -> float:
    random.shuffle(examples)
    total_loss = 0.0
    for example in examples:
        total_loss += train_one_example(network, example, config, positive_weight)
    return total_loss / len(examples)


def train_network(examples: list[TrainingExample], input_count: int, config: ModelConfig) -> tuple[NeuralNetwork, list[dict[str, object]]]:
    network = initialise_network(input_count, config)
    positive_weight = positive_class_weight(examples)
    history = []
    for epoch_number in range(1, config.epoch_count + 1):
        average_loss = train_epoch(network, examples, config, positive_weight)
        history.append(history_row(config.model_name, epoch_number, average_loss))
    return network, history


def history_row(model_name: str, epoch_number: int, average_loss: float) -> dict[str, object]:
    return {"model_name": model_name, "epoch": epoch_number, "average_loss": round(average_loss, 6)}


def predict_score(network: NeuralNetwork, example: TrainingExample, config: ModelConfig) -> float:
    _, _, score = forward_pass(network, example.features, config, False)
    return score


def predict_scores(network: NeuralNetwork, examples: list[TrainingExample], config: ModelConfig) -> list[float]:
    return [predict_score(network, example, config) for example in examples]


def predictions_from_threshold(scores: list[float], threshold_value: float) -> list[int]:
    return [1 if score >= threshold_value else 0 for score in scores]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_counts(actual_values: list[int], predicted_values: list[int]) -> dict[str, int]:
    return {
        "true_positive": sum(actual == 1 and predicted == 1 for actual, predicted in zip(actual_values, predicted_values)),
        "false_positive": sum(actual == 0 and predicted == 1 for actual, predicted in zip(actual_values, predicted_values)),
        "true_negative": sum(actual == 0 and predicted == 0 for actual, predicted in zip(actual_values, predicted_values)),
        "false_negative": sum(actual == 1 and predicted == 0 for actual, predicted in zip(actual_values, predicted_values)),
    }


def evaluate_predictions(actual_values: list[int], predicted_values: list[int]) -> dict[str, float]:
    counts = confusion_counts(actual_values, predicted_values)
    precision = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
    recall = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
    specificity = safe_divide(counts["true_negative"], counts["true_negative"] + counts["false_positive"])
    return build_metric_dictionary(counts, precision, recall, specificity)


def build_metric_dictionary(counts: dict[str, int], precision: float, recall: float, specificity: float) -> dict[str, float]:
    accuracy = safe_divide(counts["true_positive"] + counts["true_negative"], sum(counts.values()))
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2, "f1_score": f1_score}


def pairwise_auc_score(positive_scores: list[float], negative_scores: list[float]) -> float:
    comparisons = 0.0
    for positive_score in positive_scores:
        comparisons += sum(positive_score > negative_score for negative_score in negative_scores)
        comparisons += 0.5 * sum(positive_score == negative_score for negative_score in negative_scores)
    return comparisons / (len(positive_scores) * len(negative_scores))


def auc_score(actual_values: list[int], score_values: list[float]) -> float:
    positive_scores = [score for actual, score in zip(actual_values, score_values) if actual == 1]
    negative_scores = [score for actual, score in zip(actual_values, score_values) if actual == 0]
    if not positive_scores or not negative_scores:
        return 0.0
    return pairwise_auc_score(positive_scores, negative_scores)


def threshold_f1_score(actual_values: list[int], score_values: list[float], threshold_value: float) -> float:
    predicted_values = predictions_from_threshold(score_values, threshold_value)
    return evaluate_predictions(actual_values, predicted_values)["f1_score"]


def best_threshold(actual_values: list[int], score_values: list[float]) -> float:
    scored_thresholds = [(threshold, threshold_f1_score(actual_values, score_values, threshold)) for threshold in THRESHOLDS]
    return max(scored_thresholds, key=lambda item: item[1])[0]


def metric_metadata(model_name: str, actual_values: list[int], score_values: list[float], threshold_value: float) -> dict[str, object]:
    return {"model_name": model_name, "auc": auc_score(actual_values, score_values), "threshold": threshold_value, "script_version": SCRIPT_VERSION}


def build_metrics_row(model_name: str, actual_values: list[int], predictions: list[int], scores: list[float], threshold_value: float) -> dict[str, object]:
    metrics = evaluate_predictions(actual_values, predictions)
    metrics.update(metric_metadata(model_name, actual_values, scores, threshold_value))
    return order_metric_row(metrics)


def order_metric_row(metrics: dict[str, object]) -> dict[str, object]:
    ordered_names = ["model_name", "accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1_score", "auc", "threshold", "script_version"]
    return {name: metrics[name] for name in ordered_names}


def baseline_metrics(test_examples: list[TrainingExample]) -> dict[str, object]:
    actual_values = [example.target for example in test_examples]
    baseline_predictions = [0 for _ in test_examples]
    return build_metrics_row("majority_class_baseline", actual_values, baseline_predictions, baseline_predictions, 0.0)


def build_prediction_row(model_name: str, example: TrainingExample, score: float, prediction: int) -> dict[str, object]:
    return {"canonical_entity_id": example.entity_id, "model_name": model_name, "actual": example.target, "score": score, "prediction": prediction}


def build_prediction_rows(model_name: str, examples: list[TrainingExample], scores: list[float], threshold_value: float) -> list[dict[str, object]]:
    predictions = predictions_from_threshold(scores, threshold_value)
    return [build_prediction_row(model_name, example, score, prediction) for example, score, prediction in zip(examples, scores, predictions)]


def baseline_prediction_rows(test_examples: list[TrainingExample]) -> list[dict[str, object]]:
    return [build_prediction_row("majority_class_baseline", example, 0.0, 0) for example in test_examples]


def train_and_evaluate_config(config: ModelConfig, train_examples: list[TrainingExample], test_examples: list[TrainingExample], input_count: int) -> dict[str, object]:
    network, history = train_network(list(train_examples), input_count, config)
    train_scores = predict_scores(network, train_examples, config)
    threshold_value = best_threshold([example.target for example in train_examples], train_scores)
    test_scores = predict_scores(network, test_examples, config)
    return package_model_result(config, network, history, test_examples, test_scores, threshold_value)


def package_model_result(config: ModelConfig, network: NeuralNetwork, history: list[dict[str, object]], test_examples: list[TrainingExample], scores: list[float], threshold_value: float) -> dict[str, object]:
    actual_values = [example.target for example in test_examples]
    predictions = predictions_from_threshold(scores, threshold_value)
    return {"metrics": build_metrics_row(config.model_name, actual_values, predictions, scores, threshold_value), "predictions": build_prediction_rows(config.model_name, test_examples, scores, threshold_value), "history": history, "network": network}


def train_all_configs(train_examples: list[TrainingExample], test_examples: list[TrainingExample], input_count: int) -> dict[str, dict[str, object]]:
    results = {}
    for config in model_configs():
        print(f"Training {config.model_name}...")
        results[config.model_name] = train_and_evaluate_config(config, train_examples, test_examples, input_count)
    return results


def collect_metrics(results: dict[str, dict[str, object]], test_examples: list[TrainingExample]) -> list[dict[str, object]]:
    rows = [baseline_metrics(test_examples)]
    rows.extend(result["metrics"] for result in results.values())
    return rows


def collect_predictions(results: dict[str, dict[str, object]], test_examples: list[TrainingExample]) -> list[dict[str, object]]:
    rows = baseline_prediction_rows(test_examples)
    for result in results.values():
        rows.extend(result["predictions"])
    return rows


def collect_history(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for result in results.values():
        rows.extend(result["history"])
    return rows


def selected_feature_rows(feature_names: list[str]) -> list[dict[str, object]]:
    return [{"feature_name": name, "script_version": SCRIPT_VERSION} for name in feature_names]


def best_model_name(metrics_rows: list[dict[str, object]]) -> str:
    real_models = [row for row in metrics_rows if row["model_name"] != "majority_class_baseline"]
    return max(real_models, key=lambda row: row["f1_score"])["model_name"]


def comparison_report(metrics_rows: list[dict[str, object]], feature_names: list[str], input_file: Path) -> list[dict[str, object]]:
    return [
        {"metric": "input_file", "value": str(input_file)},
        {"metric": "model_count", "value": 6},
        {"metric": "feature_count", "value": len(feature_names)},
        {"metric": "best_model_by_f1", "value": best_model_name(metrics_rows)},
        {"metric": "script_version", "value": SCRIPT_VERSION},
    ]


def trained_model_payload(results: dict[str, dict[str, object]], scaling_stats: ScalingStats) -> dict[str, object]:
    return {"script_version": SCRIPT_VERSION, "scaling": asdict(scaling_stats), "models": model_payloads(results)}


def model_payloads(results: dict[str, dict[str, object]]) -> dict[str, object]:
    return {model_name: asdict(result["network"]) for model_name, result in results.items()}


def write_csv_rows(output_file: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    try:
        with output_file.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as write_error:
        raise OSError(f"Could not write {output_file}") from write_error


def write_json_file(output_file: Path, payload: dict[str, object]) -> None:
    try:
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as write_error:
        raise OSError(f"Could not write {output_file}") from write_error


def write_all_outputs(output_folder: Path, output_bundle: dict[str, object]) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    write_metrics(output_folder, output_bundle["metrics"])
    write_predictions(output_folder, output_bundle["predictions"])
    write_history(output_folder, output_bundle["history"])
    write_selected_features(output_folder, output_bundle["features"])
    write_report(output_folder, output_bundle["report"])
    write_json_file(output_folder / "dataset3_dl_trained_models.json", output_bundle["models"])


def write_metrics(output_folder: Path, rows: list[dict[str, object]]) -> None:
    fields = ["model_name", "accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1_score", "auc", "threshold", "script_version"]
    write_csv_rows(output_folder / "dataset3_dl_all_model_metrics.csv", rows, fields)


def write_predictions(output_folder: Path, rows: list[dict[str, object]]) -> None:
    fields = ["canonical_entity_id", "model_name", "actual", "score", "prediction"]
    write_csv_rows(output_folder / "dataset3_dl_all_predictions.csv", rows, fields)


def write_history(output_folder: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(output_folder / "dataset3_dl_training_history.csv", rows, ["model_name", "epoch", "average_loss"])


def write_selected_features(output_folder: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(output_folder / "dataset3_dl_selected_features.csv", rows, ["feature_name", "script_version"])


def write_report(output_folder: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(output_folder / "dataset3_dl_model_comparison_report.csv", rows, ["metric", "value"])


def build_output_bundle(results: dict[str, dict[str, object]], test_examples: list[TrainingExample], feature_names: list[str], input_file: Path, scaling_stats: ScalingStats) -> dict[str, object]:
    metrics_rows = collect_metrics(results, test_examples)
    return {"metrics": metrics_rows, "predictions": collect_predictions(results, test_examples), "history": collect_history(results), "features": selected_feature_rows(feature_names), "report": comparison_report(metrics_rows, feature_names, input_file), "models": trained_model_payload(results, scaling_stats)}


def run_deep_learning_step() -> None:
    random.seed(RANDOM_SEED)
    input_file = find_input_file()
    feature_rows = load_feature_rows(input_file)
    examples, feature_names = prepare_training_examples(feature_rows)
    split_data = make_stratified_split(examples)
    validate_split(split_data)
    run_scaled_pipeline(split_data, feature_names, input_file)


def run_scaled_pipeline(split_data: TrainTestSplit, feature_names: list[str], input_file: Path) -> None:
    scaling_stats = fit_scaling_stats(split_data.train_examples, feature_names)
    train_examples = scale_examples(split_data.train_examples, scaling_stats)
    test_examples = scale_examples(split_data.test_examples, scaling_stats)
    results = train_all_configs(train_examples, test_examples, len(feature_names))
    output_bundle = build_output_bundle(results, test_examples, feature_names, input_file, scaling_stats)
    write_all_outputs(input_file.parent, output_bundle)
    print("Step 3 complete: five Python deep learning models were trained.")


def build_sample_rows() -> list[dict[str, str]]:
    return [
        {"canonical_entity_id": "A", "artefact_count": "10", "degree_total": "50", "influence_score": "90", TARGET_COLUMN: "1"},
        {"canonical_entity_id": "B", "artefact_count": "1", "degree_total": "2", "influence_score": "1", TARGET_COLUMN: "0"},
        {"canonical_entity_id": "C", "artefact_count": "8", "degree_total": "40", "influence_score": "80", TARGET_COLUMN: "1"},
        {"canonical_entity_id": "D", "artefact_count": "0", "degree_total": "0", "influence_score": "0", TARGET_COLUMN: "0"},
    ]


def test_sunny_day_examples() -> None:
    sample_rows = build_sample_rows()
    examples, _ = prepare_training_examples(sample_rows)
    # Test 1 (Sunny Day): normal rows should become examples because training depends on them.
    assert len(examples) == 4
    print("PASS: Sunny day test passed because training examples were created.")


def test_edge_case_constant_feature() -> None:
    sample_rows = build_sample_rows()
    for sample_row in sample_rows:
        sample_row["constant_feature"] = "1"
    # Test 2 (Edge Case): constant columns are removed because they add no learning signal.
    assert "constant_feature" not in selected_feature_names(sample_rows)
    print("PASS: Edge case test passed because constant features are removed.")


def test_weird_gotcha_leakage() -> None:
    sample_rows = build_sample_rows()
    _, feature_names = prepare_training_examples(sample_rows)
    # Test 3 (Weird Gotcha): influence_score is removed because it leaks the answer.
    assert "influence_score" not in feature_names
    print("PASS: Weird gotcha test passed because leakage features are removed.")


def test_bug_catcher_metric_safety() -> None:
    actual_values = [1, 1, 0, 0]
    predicted_values = [0, 0, 0, 0]
    # Test 4 (Bug Catcher): precision should not crash when no positive cases are predicted.
    assert evaluate_predictions(actual_values, predicted_values)["precision"] == 0
    print("PASS: Bug catcher test passed because zero-positive predictions are safe.")


def run_self_tests() -> None:
    test_sunny_day_examples()
    test_edge_case_constant_feature()
    test_weird_gotcha_leakage()
    test_bug_catcher_metric_safety()


if __name__ == "__main__":
    run_self_tests()
    run_deep_learning_step()
    
