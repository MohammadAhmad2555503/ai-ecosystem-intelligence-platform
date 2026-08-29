"""
A/B test statistics for Step 6.

The primary metric is positive feedback rate:
rating >= 4 counts as a successful answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VariantSummary:
    model_name: str
    sample_size: int
    success_count: int
    success_rate: float


@dataclass
class AbTestResult:
    primary_metric: str
    baseline: VariantSummary
    challenger: VariantSummary
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval_low: float
    confidence_interval_high: float
    effect_size: float
    required_sample_per_variant: int
    observed_power: float
    decision: str


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def z_score_for_confidence(confidence_level: float) -> float:
    if confidence_level >= 0.99:
        return 2.576
    if confidence_level >= 0.95:
        return 1.96
    return 1.645


def safe_rate(success_count: int, sample_size: int) -> float:
    if sample_size == 0:
        return 0.0
    return success_count / sample_size


def pooled_rate(left_successes: int, right_successes: int, total_size: int) -> float:
    if total_size == 0:
        return 0.0
    return (left_successes + right_successes) / total_size


def pooled_standard_error(
    pooled_success_rate: float,
    baseline_size: int,
    challenger_size: int,
) -> float:
    if baseline_size == 0 or challenger_size == 0:
        return 0.0
    variance = pooled_success_rate * (1 - pooled_success_rate)
    variance *= (1 / baseline_size) + (1 / challenger_size)
    return math.sqrt(max(variance, 0.0))


def standard_error_for_difference(
    left_rate: float,
    right_rate: float,
    left_size: int,
    right_size: int,
) -> float:
    if left_size == 0 or right_size == 0:
        return 0.0
    variance = left_rate * (1 - left_rate) / left_size
    variance += right_rate * (1 - right_rate) / right_size
    return math.sqrt(max(variance, 0.0))


def two_proportion_p_value(baseline: VariantSummary, challenger: VariantSummary) -> float:
    total_size = baseline.sample_size + challenger.sample_size
    combined_rate = pooled_rate(baseline.success_count, challenger.success_count, total_size)
    standard_error = pooled_standard_error(
        combined_rate,
        baseline.sample_size,
        challenger.sample_size,
    )
    if standard_error == 0:
        return 1.0
    z_value = (challenger.success_rate - baseline.success_rate) / standard_error
    return 2.0 * (1.0 - normal_cdf(abs(z_value)))


def confidence_interval(
    baseline: VariantSummary,
    challenger: VariantSummary,
    confidence_level: float,
) -> tuple[float, float]:
    lift = challenger.success_rate - baseline.success_rate
    error = standard_error_for_difference(
        challenger.success_rate,
        baseline.success_rate,
        challenger.sample_size,
        baseline.sample_size,
    )
    z_value = z_score_for_confidence(confidence_level)
    return lift - z_value * error, lift + z_value * error


def bound_probability(value: float) -> float:
    return max(0.0, min(value, 1.0))


def cohen_h(left_rate: float, right_rate: float) -> float:
    left_angle = math.asin(math.sqrt(bound_probability(left_rate)))
    right_angle = math.asin(math.sqrt(bound_probability(right_rate)))
    return 2.0 * (right_angle - left_angle)


def approximate_z_for_power(power: float) -> float:
    if power >= 0.9:
        return 1.282
    if power >= 0.8:
        return 0.842
    return 0.524


def rate_variance_sum(baseline_rate: float, challenger_rate: float) -> float:
    baseline_variance = baseline_rate * (1 - baseline_rate)
    challenger_variance = challenger_rate * (1 - challenger_rate)
    return math.sqrt(baseline_variance + challenger_variance)


def sample_size_numerator(
    pooled_rate_value: float,
    baseline_rate: float,
    challenger_rate: float,
    alpha: float,
    power: float,
) -> float:
    z_alpha = z_score_for_confidence(1 - alpha)
    z_power = approximate_z_for_power(power)
    first_term = z_alpha * math.sqrt(2 * pooled_rate_value * (1 - pooled_rate_value))
    second_term = z_power * rate_variance_sum(baseline_rate, challenger_rate)
    return (first_term + second_term) ** 2


def required_sample_per_variant(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float,
    power: float,
) -> int:
    challenger_rate = bound_probability(baseline_rate + minimum_detectable_effect)
    pooled_value = (baseline_rate + challenger_rate) / 2.0
    numerator = sample_size_numerator(pooled_value, baseline_rate, challenger_rate, alpha, power)
    denominator = max((challenger_rate - baseline_rate) ** 2, 0.000001)
    return math.ceil(numerator / denominator)


def observed_power(baseline: VariantSummary, challenger: VariantSummary, alpha: float) -> float:
    effect = abs(challenger.success_rate - baseline.success_rate)
    standard_error = standard_error_for_difference(
        baseline.success_rate,
        challenger.success_rate,
        baseline.sample_size,
        challenger.sample_size,
    )
    if standard_error == 0:
        return 0.0
    critical_value = z_score_for_confidence(1 - alpha)
    return normal_cdf((effect / standard_error) - critical_value)


def make_decision(p_value: float, lift: float, power_value: float) -> str:
    if p_value < 0.05 and lift > 0 and power_value >= 0.8:
        return "ship"
    if p_value < 0.1 and lift > 0:
        return "iterate"
    return "hold"


def relative_lift_value(baseline_rate: float, absolute_lift: float) -> float:
    if baseline_rate == 0:
        return 0.0
    return absolute_lift / baseline_rate


def run_two_proportion_test(
    baseline: VariantSummary,
    challenger: VariantSummary,
    minimum_detectable_effect: float = 0.1,
) -> AbTestResult:
    interval = confidence_interval(baseline, challenger, 0.95)
    p_value = two_proportion_p_value(baseline, challenger)
    metrics = computed_metrics(
        baseline,
        challenger,
        minimum_detectable_effect,
        p_value,
        interval,
    )
    return make_result_object(baseline, challenger, metrics)


def computed_metrics(
    baseline: VariantSummary,
    challenger: VariantSummary,
    minimum_detectable_effect: float,
    p_value: float,
    interval: tuple[float, float],
) -> dict[str, float]:
    lift = challenger.success_rate - baseline.success_rate
    power_value = observed_power(baseline, challenger, 0.05)
    required_size = sample_requirement(baseline, minimum_detectable_effect)
    values = (lift, p_value, interval[0], interval[1], power_value, required_size)
    return result_metric_dict(values)


def sample_requirement(baseline: VariantSummary, minimum_detectable_effect: float) -> int:
    return required_sample_per_variant(
        baseline.success_rate,
        minimum_detectable_effect,
        0.05,
        0.8,
    )


def result_metric_dict(values: tuple[float, float, float, float, float, int]) -> dict[str, float]:
    return {
        "lift": values[0],
        "p_value": values[1],
        "low_value": values[2],
        "high_value": values[3],
        "power_value": values[4],
        "required_size": float(values[5]),
    }


def make_result_object(
    baseline: VariantSummary,
    challenger: VariantSummary,
    metrics: dict[str, float],
) -> AbTestResult:
    lift = metrics["lift"]
    return AbTestResult(*result_values(baseline, challenger, metrics, lift))


def result_values(
    baseline: VariantSummary,
    challenger: VariantSummary,
    metrics: dict[str, float],
    lift: float,
) -> tuple[object, ...]:
    return (
        "positive_feedback_rate", baseline, challenger, lift,
        relative_lift_value(baseline.success_rate, lift),
        metrics["p_value"], metrics["low_value"], metrics["high_value"],
        cohen_h(baseline.success_rate, challenger.success_rate),
        int(metrics["required_size"]), metrics["power_value"],
        make_decision(metrics["p_value"], lift, metrics["power_value"]),
    )


